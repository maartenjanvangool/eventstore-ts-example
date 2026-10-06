/**
 * Regression tests for the append-condition race (audit Finding 3).
 *
 * Before the fix, `append()` ran its CTE guard
 * (src/eventstore/postgres.ts:114-131) as a bare autocommit statement. The
 * guard reads a statement-start snapshot and locks nothing it aggregated, so two
 * writers holding the same `expectedMaxSequence` both passed it under READ
 * COMMITTED. Measured on 2026-10-02 before the fix:
 *
 *   - staged two sessions, same expected position  -> both inserted, 2 rows
 *   - store.append() x2, 30 rounds               -> 22-29 rounds lost the guard
 *   - two concurrent openBankAccount()           -> 0-8 of 25 rounds created
 *     two accounts for one customer name (rate depended on table size, i.e.
 *     the unstaged test could pass without the bug firing)
 *
 * The fix (src/eventstore/postgres.ts, `append()` + `contextLockKeys()`): take
 * a `pg_advisory_xact_lock` per event type in the filter, in sorted order,
 * inside an explicit transaction, before the CTE runs. The guard is then
 * evaluated against a snapshot that contains every previously committed write
 * to that context.
 *
 * Tests 1, 4, 5 and 6 stage the interleaving and are deterministic. Tests 2 and 3
 * are true-concurrency rounds; after the fix they assert zero lost updates,
 * which the lock makes impossible rather than unlikely.
 *
 * Needs a running PostgreSQL, same as the existing tests:
 *   docker run -d --name es-race-pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16
 *   DATABASE_URL=postgres://postgres:postgres@localhost:5432/eventstore_race_test \
 *     npx jest tests/concurrent-append-race.test.ts
 */

import { Client } from 'pg';
import dotenv from 'dotenv';
import { EventStore, EventFilter } from '../src/eventstore';
import { HasEventType } from '../src/eventstore/types';
import { execute as openBankAccount } from '../src/features/open-bank-account';
import { OpenAccountError, OpenAccountResult } from '../src/features/open-bank-account/types';
import { BankAccountOpenedEvent } from '../src/features/open-bank-account/events';

dotenv.config();

// Own database: jest runs suites in parallel workers against the same Postgres,
// and tests 1, 4 and 5 depend on who else writes to `events`.
const CONNECTION =
  process.env.DATABASE_RACE_TEST_URL ||
  'postgres://postgres:postgres@localhost:5432/eventstore_race_test';

// PostgresEventStore.migrate() creates the database from process.env.DATABASE_URL
// and ignores the connectionString given to the constructor
// (src/eventstore/postgres.ts:186-189), so point both at the test database.
process.env.DATABASE_URL = process.env.DATABASE_URL || CONNECTION;

class RaceEvent implements HasEventType {
  constructor(
    public readonly id: string,
    public readonly writer: string,
    public readonly eventTypeName: string,
  ) {}
  eventType(): string {
    return this.eventTypeName;
  }
  eventVersion(): string {
    return '1.0';
  }
}

type StoreInternals = {
  buildCteInsertQuery(filter: EventFilter, expectedMaxSeq: number): string;
  buildFilterCondition(filter: EventFilter): { condition: string; params: unknown[] };
  contextLockKeys(filter: EventFilter): string[];
  contextLockStatement(key: string): { text: string; params: unknown[] };
};

const internals = (store: EventStore): StoreInternals => store as unknown as StoreInternals;

/**
 * The store's own append SQL for `filter` / `expectedMaxSequence`, with the
 * parameter layout `append()` uses (src/eventstore/postgres.ts:59-70). Reached
 * through the private builder so the test cannot drift from the SQL the store
 * actually runs.
 */
function appendSql(
  store: EventStore,
  filter: EventFilter,
  expectedMaxSequence: number,
  makeEvent: (tag: string) => HasEventType,
): { sql: string; params: (tag: string) => unknown[] } {
  const filterParams = internals(store).buildFilterCondition(filter).params;
  return {
    sql: internals(store).buildCteInsertQuery(filter, expectedMaxSequence),
    params: (tag: string) => {
      const event = makeEvent(tag);
      return [
        ...filterParams,
        [event.eventType()],
        [JSON.stringify(event)],
        [JSON.stringify({ version: event.eventVersion?.() || '1.0' })],
      ];
    },
  };
}

async function newClient(isolation?: 'serializable'): Promise<Client> {
  const client = new Client({ connectionString: CONNECTION });
  await client.connect();
  if (isolation) {
    await client.query('SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL SERIALIZABLE');
  }
  return client;
}

async function rowCount(store: EventStore, filter: EventFilter): Promise<number> {
  const { events } = await store.query<HasEventType>(filter);
  return events.length;
}

async function totalRows(): Promise<number> {
  const c = await newClient();
  try {
    const r = await c.query('SELECT count(*)::int AS n FROM events');
    return r.rows[0].n;
  } finally {
    await c.end();
  }
}

const raceEvent = (eventType: string) => (tag: string): RaceEvent =>
  new RaceEvent(`race-${tag}`, tag, eventType);

describe('Append-condition race (audit Finding 3)', () => {
  let store: EventStore;

  beforeEach(async () => {
    store = new EventStore({ connectionString: CONNECTION });
    await store.migrate();
  });

  afterEach(async () => {
    await store.close();
  });

  it('test 1 (deterministic): a held context lock makes the second writer see the first', async () => {
    const eventType = `LockDet_${Date.now()}`;
    const filter = EventFilter.createFilter([eventType]);
    const { sql, params } = appendSql(store, filter, 0, raceEvent(eventType));
    const lock = internals(store).contextLockStatement(eventType);

    expect(await rowCount(store, filter)).toBe(0);

    const a = await newClient();
    const b = await newClient();
    try {
      // Writer A enters the context and holds the lock open.
      await a.query('BEGIN');
      await a.query(lock.text, lock.params);
      const insertA = await a.query(sql, params('A'));
      expect(insertA.rowCount).toBe(1);

      // Writer B tries to enter the same context while A holds it.
      let bFinished = false;
      const bRun = (async () => {
        await b.query('BEGIN');
        await b.query(lock.text, lock.params); // blocks here until A commits
        const r = await b.query(sql, params('B'));
        await b.query('COMMIT');
        return r;
      })().then((r) => {
        bFinished = true;
        return r;
      });

      await new Promise((resolve) => setTimeout(resolve, 500));
      expect(bFinished).toBe(false); // blocked on the lock, not racing past it

      await a.query('COMMIT');
      const insertB = await bRun;

      // B's guard now runs against a snapshot that includes A's row: rejected.
      expect(insertB.rowCount).toBe(0);
      expect(await rowCount(store, filter)).toBe(1);
    } finally {
      await a.end();
      await b.end();
    }
  });

  it('test 2 (public API, stress): concurrent store.append() at the same position loses no updates', async () => {
    const ROUNDS = 30;
    let lostUpdateRounds = 0;
    const acceptedCounts = new Map<number, number>();

    for (let round = 0; round < ROUNDS; round++) {
      const eventType = `RaceApi_${Date.now()}_${round}`;
      const filter = EventFilter.createFilter([eventType]);

      const results = await Promise.allSettled([
        store.append(filter, [raceEvent(eventType)('A')], 0),
        store.append(filter, [raceEvent(eventType)('B')], 0),
      ]);

      const accepted = results.filter((r) => r.status === 'fulfilled').length;
      const inserted = await rowCount(store, filter);
      acceptedCounts.set(accepted, (acceptedCounts.get(accepted) ?? 0) + 1);

      if (inserted > 1) {
        lostUpdateRounds++;
        console.log(`  round ${round}: appends accepted=${accepted}, rows inserted=${inserted}`);
      }
    }

    console.log(`[outcome] accepted-per-round: ${JSON.stringify([...acceptedCounts])}`);
    console.log(`[race rate] ${lostUpdateRounds}/${ROUNDS} rounds lost the append-condition`);

    // Exactly one writer may win each round: the loser is rejected by the guard,
    // not silently allowed through.
    expect(lostUpdateRounds).toBe(0);
    expect(acceptedCounts.get(1)).toBe(ROUNDS);
  }, 120_000);

  it('test 3 (domain, stress): concurrent openBankAccount() never creates two accounts for one name', async () => {
    const ROUNDS = 25;
    let duplicateRounds = 0;
    let exactlyOneWinner = 0;
    let rejectedByUniqueness = 0;
    let rejectedByGuard = 0;

    const isRejected = (
      r: OpenAccountResult,
    ): r is { success: false; error: OpenAccountError } => !r.success;

    for (let round = 0; round < ROUNDS; round++) {
      const customerName = `Alice-${Date.now()}-${round}`;

      const results = await Promise.all([
        openBankAccount(store, { customerName, accountType: 'checking', initialDeposit: 100, currency: 'USD' }),
        openBankAccount(store, { customerName, accountType: 'savings', initialDeposit: 200, currency: 'EUR' }),
      ]);

      const accounts = await store.query<HasEventType>(
        EventFilter.new(['BankAccountOpened']).withPayloadPredicates({ customerName }),
      );

      const successes = results.filter((r) => r.success).length;

      if (accounts.events.length > 1) {
        duplicateRounds++;
        console.log(`  round ${round}: ${accounts.events.length} accounts for ${customerName}`);
      } else if (successes === 1) {
        exactlyOneWinner++;
        const loser = results.find(isRejected);
        if (loser?.error.message === 'Customer name already exists') {
          rejectedByUniqueness++; // the read already saw the committed name
        } else {
          rejectedByGuard++; // the read missed it; the append-condition caught it
        }
      }
    }

    console.log(
      `[outcome] one winner in ${exactlyOneWinner}/${ROUNDS} rounds; ` +
        `loser rejected by uniqueness check=${rejectedByUniqueness}, ` +
        `by append guard=${rejectedByGuard}`,
    );
    console.log(`[race rate] ${duplicateRounds}/${ROUNDS} rounds opened duplicate accounts`);

    expect(duplicateRounds).toBe(0);
    expect(exactlyOneWinner).toBe(ROUNDS);
  }, 120_000);

  it('test 4 (deterministic, domain): a concurrent "Alice" that commits mid-flight rejects the shell', async () => {
    const customerName = `Alice-Det-${Date.now()}`;
    const openedFilter = EventFilter.createFilter(['BankAccountOpened']);
    const { maxSequenceNumber } = await store.query<HasEventType>(openedFilter);
    const { sql, params } = appendSql(
      store,
      openedFilter,
      maxSequenceNumber,
      (tag) => new BankAccountOpenedEvent(`concurrent-${tag}`, customerName, 'savings', 200, 'EUR'),
    );

    const a = await newClient();
    try {
      // Writer A holds the BankAccountOpened context lock with an "Alice" written
      // but not yet committed.
      await a.query('BEGIN');
      for (const key of internals(store).contextLockKeys(openedFilter)) {
        const lock = internals(store).contextLockStatement(key);
        await a.query(lock.text, lock.params);
      }
      const insertA = await a.query(sql, params('B'));
      expect(insertA.rowCount).toBe(1);

      // The shell's query() cannot see A's uncommitted row, so its uniqueness
      // check passes and it reaches the append — where it must now wait.
      let shellFinished = false;
      const shellRun = openBankAccount(store, {
        customerName,
        accountType: 'checking',
        initialDeposit: 100,
        currency: 'USD',
      }).then((r) => {
        shellFinished = true;
        return r;
      });

      await new Promise((resolve) => setTimeout(resolve, 500));
      expect(shellFinished).toBe(false); // blocked on the lock, not through the guard

      await a.query('COMMIT');
      const result = await shellRun;

      // A's write landed first; the shell's guard now sees it and rejects.
      expect(result.success).toBe(false);
      const accounts = await store.query<HasEventType>(
        EventFilter.new(['BankAccountOpened']).withPayloadPredicates({ customerName }),
      );
      expect(accounts.events).toHaveLength(1);
      expect((accounts.events[0] as unknown as { accountId: string }).accountId).toBe('concurrent-B');
    } finally {
      await a.end();
    }
  });

  it('test 5 (lock scoping): a held lock on one event type does not block another', async () => {
    const heldType = `ScopeHeld_${Date.now()}`;
    const freeType = `ScopeFree_${Date.now()}`;
    const lock = internals(store).contextLockStatement(heldType);

    const holder = await newClient();
    try {
      await holder.query('BEGIN');
      await holder.query(lock.text, lock.params);
      await holder.query('SELECT pg_sleep(0.5)'); // hold the held-type context ~500ms

      const started = Date.now();
      await store.append(
        EventFilter.createFilter([freeType]),
        [new RaceEvent('free', 'free', freeType)],
        0,
      );
      const elapsed = Date.now() - started;
      console.log(`[lock scoping] unrelated append took ${elapsed}ms while ${heldType} was held`);

      // The unrelated append must not have waited for the held lock.
      expect(elapsed).toBeLessThan(400);
      expect(await rowCount(store, EventFilter.createFilter([freeType]))).toBe(1);
    } finally {
      await holder.query('ROLLBACK');
      await holder.end();
    }
  });

  it('test 6 (control): SERIALIZABLE also catches the race, and the loser burns a sequence number', async () => {
    const eventType = `RaceSer_${Date.now()}`;
    const filter = EventFilter.createFilter([eventType]);
    const { sql, params } = appendSql(store, filter, 0, raceEvent(eventType));

    const seqBefore = await sequenceValue();
    const rowsBefore = await totalRows();

    const a = await newClient('serializable');
    const b = await newClient('serializable');
    let aCommitError: (Error & { code?: string }) | null = null;
    let bCommitError: (Error & { code?: string }) | null = null;

    try {
      await a.query('BEGIN');
      const insertA = await a.query(sql, params('A'));
      expect(insertA.rowCount).toBe(1);

      const bDone = (async () => {
        await b.query('BEGIN');
        await b.query(sql, params('B'));
        await b.query('COMMIT');
      })().catch((err: Error & { code?: string }) => err);

      // Let B evaluate its guard against A's pre-commit snapshot.
      await new Promise((resolve) => setTimeout(resolve, 1500));

      try {
        await a.query('COMMIT');
      } catch (err) {
        aCommitError = err as Error & { code?: string };
      }
      const bErr = await bDone;
      bCommitError = bErr instanceof Error ? bErr : null;

      const committed = (aCommitError ? 0 : 1) + (bCommitError ? 0 : 1);
      const loser = aCommitError ?? bCommitError;

      // Exactly one writer lands...
      expect(committed).toBe(1);
      expect(await rowCount(store, filter)).toBe(1);
      // ...and SSI rejected the other, not the CTE guard.
      expect(loser).not.toBeNull();
      expect(loser!.code).toBe('40001'); // serialization_failure

      // Both transactions consumed a nextval; only one row landed, so at least
      // one sequence value was burned by the abort (nextval is not transactional:
      // https://www.postgresql.org/docs/current/functions-sequence.html).
      // Asserted as >= 1, not == 1, so other suites writing concurrently cannot
      // turn this into a false failure.
      const seqAfter = await sequenceValue();
      const rowsAfter = await totalRows();
      const burned = seqAfter - seqBefore - (rowsAfter - rowsBefore);
      expect(burned).toBeGreaterThanOrEqual(1);
    } finally {
      await a.end();
      await b.end();
    }

    async function sequenceValue(): Promise<number> {
      const c = await newClient();
      try {
        const r = await c.query('SELECT last_value::int AS v FROM events_sequence_number_seq');
        return r.rows[0].v;
      } finally {
        await c.end();
      }
    }
  }, 60_000);
});
