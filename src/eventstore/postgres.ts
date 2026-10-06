import { Pool } from 'pg';
import { IEventStore, EventFilter, HasEventType, EventRecord, EventStoreOptions } from './types';
import { EventFilter as EventFilterClass } from './filter';

export class PostgresEventStore implements IEventStore {
  private pool: Pool;
  private readonly dbName: string;

  constructor(options: EventStoreOptions = {}) {
    const connectionString = options.connectionString || process.env.DATABASE_URL;
    
    if (!connectionString) {
      throw new Error('DATABASE_URL environment variable is required');
    }

    this.dbName = this.getDatabaseNameFromConnectionString(connectionString) || 'bank';
    this.pool = new Pool({ connectionString });
  }

  async query<T extends HasEventType>(filter: EventFilter): Promise<{ events: T[]; maxSequenceNumber: number }> {
    const client = await this.pool.connect();
    try {
      const filterCondition = this.buildFilterCondition(filter);
      
      const query = `
        SELECT *, 
               MAX(sequence_number) OVER () AS max_seq_overall
        FROM events 
        WHERE ${filterCondition.condition}
        ORDER BY sequence_number ASC
      `;

      const result = await client.query(query, filterCondition.params);
      
      const events = result.rows.map(row => this.deserializeEvent<T>(row));
      const maxSequenceNumber = result.rows.length > 0 ? 
        parseInt(result.rows[0].max_seq_overall, 10) : 0;
      
      return { events, maxSequenceNumber };
    } finally {
      client.release();
    }
  }

  async append<T extends HasEventType>(filter: EventFilter, events: T[], expectedMaxSequence: number): Promise<void> {
    if (events.length === 0) return;

    const client = await this.pool.connect();
    try {

      const eventTypes: string[] = [];
      const payloads: string[] = [];
      const metadata: string[] = [];

      for (const event of events) {
        eventTypes.push(event.eventType());
        payloads.push(JSON.stringify(event));
        metadata.push(JSON.stringify({
          version: event.eventVersion?.() || '1.0'
        }));
      }

      const filterCondition = this.buildFilterCondition(filter);
      const cteQuery = this.buildCteInsertQuery(filter, expectedMaxSequence);
      
      const params = [
        ...filterCondition.params,                           // Filter parameters (dynamic based on filter)
        eventTypes,                                          // Event types to insert
        payloads,                                            // Payloads to insert
        metadata                                             // Metadata to insert
      ];

      // The CTE guard reads a statement-start snapshot and takes no lock on what
      // it aggregated, so two writers holding the same expectedMaxSequence both
      // pass it under READ COMMITTED. Serialize writers of the same context
      // first: the advisory lock is held until COMMIT, so the guard below is
      // evaluated against a snapshot that contains every previously committed
      // write to this context.
      await client.query('BEGIN');
      try {
        for (const key of this.contextLockKeys(filter)) {
          const lock = this.contextLockStatement(key);
          await client.query(lock.text, lock.params);
        }

        const result = await client.query(cteQuery, params);

        if (result.rowCount === 0) {
          throw new Error('Context changed: events were modified between query and append');
        }

        await client.query('COMMIT');
      } catch (error) {
        try {
          await client.query('ROLLBACK');
        } catch {
          // The connection is gone; the original error is the one worth throwing.
        }
        throw error;
      }
    } finally {
      client.release();
    }
  }

  /**
   * Advisory-lock keys for the append-condition.
   *
   * The guard aggregates `MAX(sequence_number) WHERE event_type = ANY(...) AND
   * <payload predicates>`. One append can only invalidate another append's guard
   * if their filtered row sets overlap. Two filters whose *event type* sets are
   * disjoint can never share rows, whatever their payload predicates say, so one
   * lock per event type is a sound over-approximation: it may serialize writers
   * that did not need serializing, never the other way round.
   *
   * Keys are taken in sorted order, so two appends can never hold-and-wait in
   * opposite order on the same pair of keys — no deadlock.
   */
  private contextLockKeys(filter: EventFilter): string[] {
    return [...new Set(filter.eventTypes)].sort();
  }

  /**
   * The lock statement for one key. `hashtext()` collisions only cause spurious
   * contention between unrelated contexts, never spurious concurrency, which is
   * the safe direction for a correctness lock.
   */
  private contextLockStatement(key: string): { text: string; params: unknown[] } {
    return { text: 'SELECT pg_advisory_xact_lock(hashtext($1))', params: [key] };
  }

  private buildFilterCondition(filter: EventFilter): { condition: string; params: unknown[] } {
    let condition = 'event_type = ANY($1)';
    const params: unknown[] = [filter.eventTypes];

    if (filter.payloadPredicates && Object.keys(filter.payloadPredicates).length > 0) {
      condition += ' AND payload @> $2';
      params.push(JSON.stringify(filter.payloadPredicates));
    }

    if (filter.payloadPredicateOptions && filter.payloadPredicateOptions.length > 0) {
      const orConditions = filter.payloadPredicateOptions.map((_, index) => {
        const paramIndex = params.length + 1;
        params.push(JSON.stringify(filter.payloadPredicateOptions![index]));
        return `payload @> $${paramIndex}`;
      });
      condition += ` AND (${orConditions.join(' OR ')})`;
    }

    return { condition, params };
  }

  private buildCteInsertQuery(filter: EventFilter, expectedMaxSeq: number): string {
    const filterCondition = this.buildFilterCondition(filter);

    const contextParamCount = filterCondition.params.length;
    const eventTypesParam = contextParamCount + 1;
    const payloadsParam = contextParamCount + 2;
    const metadataParam = contextParamCount + 3;

    return `
      WITH context AS (
        SELECT MAX(sequence_number) AS max_seq
        FROM events
        WHERE ${filterCondition.condition}
      )
      INSERT INTO events (event_type, payload, metadata)
      SELECT unnest($${eventTypesParam}::text[]), unnest($${payloadsParam}::jsonb[]), unnest($${metadataParam}::jsonb[])
      FROM context
      WHERE COALESCE(max_seq, 0) = ${expectedMaxSeq}
    `;
  }

  private deserializeEvent<T extends HasEventType>(row: any): T {
    const payload = typeof row.payload === 'string' ? JSON.parse(row.payload) : row.payload;
    return {
      ...payload,
      event_type: row.event_type,
      sequenceNumber: row.sequence_number,
      occurredAt: row.occurred_at
    } as T;
  }

  async close(): Promise<void> {
    await this.pool.end();
  }

  async migrate(): Promise<void> {
    await this.createTables();
  }

  async createTables(): Promise<void> {
    await this.ensureDatabaseExists();

    const client = await this.pool.connect();
    try {
      await client.query(`
        CREATE TABLE IF NOT EXISTS events (
          sequence_number BIGSERIAL PRIMARY KEY,
          occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          event_type TEXT NOT NULL,
          payload JSONB NOT NULL,
          metadata JSONB NOT NULL DEFAULT '{}'
        )
      `);

      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)
      `);

      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at)
      `);

      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_events_payload_gin ON events USING gin(payload)
      `);
    } finally {
      client.release();
    }
  }

  private async ensureDatabaseExists(): Promise<void> {
    const adminConnectionString = this.changeDatabaseInConnectionString(
      process.env.DATABASE_URL!, 
      'postgres'
    );
    
    const adminPool = new Pool({ connectionString: adminConnectionString });
    const client = await adminPool.connect();
    
    try {
      await client.query(`CREATE DATABASE ${this.dbName}`);
      console.log(`Database created: ${this.dbName}`);
    } catch (err: any) {
      if (err.code === '42P04') { // already exists
        console.log(`Database already exists: ${this.dbName}`);
      } else {
        throw err;
      }
    } finally {
      client.release();
      await adminPool.end();
    }
  }

  private changeDatabaseInConnectionString(connStr: string, newDbName: string): string {
    const url = new URL(connStr);
    url.pathname = `/${newDbName}`;
    return url.toString();
  }

  private getDatabaseNameFromConnectionString(connStr: string): string | null {
    try {
      const url = new URL(connStr);
      const dbName = url.pathname.startsWith('/') ? url.pathname.slice(1) : url.pathname;
      return dbName || null;
    } catch (err) {
      console.error('Invalid connection string:', err);
      return null;
    }
  }
  
}