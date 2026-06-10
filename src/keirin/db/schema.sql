-- Keirin SQLite schema. Idempotent: safe to run on init.
-- All timestamps are ISO 8601 strings in Asia/Tokyo unless stated.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Masters
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS venues (
    venue_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    bank_length INTEGER NOT NULL,
    bank_angle  REAL
);

CREATE TABLE IF NOT EXISTS players (
    player_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    pref        TEXT,
    birth_date  TEXT,
    style       TEXT,
    rank_class  TEXT,
    rating      REAL,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS player_history (
    player_id   TEXT NOT NULL,
    period      TEXT NOT NULL,
    rating      REAL,
    rank_class  TEXT,
    PRIMARY KEY (player_id, period),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

-- ---------------------------------------------------------------------------
-- Races and entries
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS races (
    race_id     TEXT PRIMARY KEY,
    date        TEXT NOT NULL,
    venue_id    TEXT NOT NULL,
    race_no     INTEGER NOT NULL,
    grade       TEXT,
    race_class  TEXT,
    distance_m  INTEGER,
    weather     TEXT,
    track_cond  TEXT,
    post_time   TEXT,
    FOREIGN KEY (venue_id) REFERENCES venues(venue_id),
    UNIQUE (date, venue_id, race_no)
);

CREATE TABLE IF NOT EXISTS entries (
    race_id     TEXT NOT NULL,
    car_no      INTEGER NOT NULL,
    player_id   TEXT NOT NULL,
    rank_class  TEXT,
    rating      REAL,
    gear_ratio  REAL,
    line_id     INTEGER,
    line_pos    INTEGER,
    style       TEXT,
    scratched   INTEGER DEFAULT 0,
    -- 出走表の直近4ヶ月集計値 (netkeirin 掲載時点の値 = 予測時点で入手可能)
    nige_cnt    INTEGER,        -- 逃げ回数
    makuri_cnt  INTEGER,        -- 捲り回数
    sashi_cnt   INTEGER,        -- 差し回数
    mark_cnt    INTEGER,        -- マーク回数
    s_count     INTEGER,        -- S回数 (スタート取り)
    b_count     INTEGER,        -- B回数 (バック先頭 = 先行力の直接指標)
    line_raw    TEXT,           -- ライン欄の生文字列 (パーサ診断用)
    PRIMARY KEY (race_id, car_no),
    FOREIGN KEY (race_id) REFERENCES races(race_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE TABLE IF NOT EXISTS results (
    race_id     TEXT NOT NULL,
    car_no      INTEGER NOT NULL,
    finish      INTEGER,
    kimarite    TEXT,
    time_sec    REAL,
    PRIMARY KEY (race_id, car_no),
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

CREATE TABLE IF NOT EXISTS payouts (
    race_id     TEXT NOT NULL,
    bet_type    TEXT NOT NULL,
    combo       TEXT NOT NULL,
    payout_yen  INTEGER,
    popularity  INTEGER,
    PRIMARY KEY (race_id, bet_type, combo),
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

-- ---------------------------------------------------------------------------
-- Odds (time-series snapshots)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS odds_trifecta (
    race_id     TEXT NOT NULL,
    snapshot_at TEXT NOT NULL,
    combo       TEXT NOT NULL,
    odds        REAL NOT NULL,
    PRIMARY KEY (race_id, snapshot_at, combo),
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

-- ---------------------------------------------------------------------------
-- Per-race history for each player (scraped from profile page or derived).
-- One row per player per race they participated in.
-- Used for rolling-window feature computation (last-N races, rest days, etc.)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS player_race_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id   TEXT NOT NULL,
    race_id     TEXT NOT NULL,
    race_date   TEXT NOT NULL,
    venue_id    TEXT,
    grade       TEXT,
    race_class  TEXT,
    bank_length INTEGER,
    car_no      INTEGER,
    finish      INTEGER,         -- NULL if scratched or DNF
    kimarite    TEXT,            -- 逃/捲/差/マーク
    time_sec    REAL,
    line_pos    INTEGER,         -- 1=先頭(先行), 2=番手, 3=三番手
    scratched   INTEGER DEFAULT 0,
    UNIQUE (player_id, race_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE INDEX IF NOT EXISTS idx_prl_player_date ON player_race_log(player_id, race_date);
CREATE INDEX IF NOT EXISTS idx_prl_player_venue ON player_race_log(player_id, venue_id);

-- ---------------------------------------------------------------------------
-- User bets (real or paper-trade; stake_yen=0 means paper)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS bets (
    bet_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id     TEXT NOT NULL,
    combo       TEXT NOT NULL,
    stake_yen   INTEGER NOT NULL,
    pred_prob   REAL,
    pred_ev     REAL,
    placed_at   TEXT NOT NULL,
    payout_yen  INTEGER,
    hit         INTEGER,
    is_paper    INTEGER DEFAULT 0,
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

-- ---------------------------------------------------------------------------
-- Audit logs
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS fetch_log (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT NOT NULL,
    status      INTEGER,
    fetched_at  TEXT NOT NULL,
    bytes       INTEGER
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_races_date         ON races(date);
CREATE INDEX IF NOT EXISTS idx_races_date_venue   ON races(date, venue_id);
CREATE INDEX IF NOT EXISTS idx_entries_player     ON entries(player_id);
CREATE INDEX IF NOT EXISTS idx_results_finish     ON results(race_id, finish);
CREATE INDEX IF NOT EXISTS idx_odds_race          ON odds_trifecta(race_id);
CREATE INDEX IF NOT EXISTS idx_odds_snapshot      ON odds_trifecta(race_id, snapshot_at);
CREATE INDEX IF NOT EXISTS idx_bets_race          ON bets(race_id);
CREATE INDEX IF NOT EXISTS idx_bets_placed_at     ON bets(placed_at);
CREATE INDEX IF NOT EXISTS idx_fetch_log_url      ON fetch_log(url, fetched_at);

-- ---------------------------------------------------------------------------
-- Simulation feature cache (ライン力学・展開モンテカルロの出力)
-- precompute_sim_features.py が一括計算して書き込み、builder が学習時に参照する。
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sim_features (
    race_id     TEXT NOT NULL,
    car_no      INTEGER NOT NULL,
    p_sim_win   REAL,            -- シミュレーション P(1着)
    p_sim_top3  REAL,            -- シミュレーション P(3着内)
    p_front     REAL,            -- 自ラインが主導権(先行)を取る確率
    sim_version TEXT,
    PRIMARY KEY (race_id, car_no),
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

-- ---------------------------------------------------------------------------
-- Prediction log (for feedback loop and accuracy tracking)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS prediction_log (
    log_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id             TEXT NOT NULL,
    predicted_at        TEXT NOT NULL,
    pred_rank1_car      INTEGER,
    pred_rank2_car      INTEGER,
    pred_rank3_car      INTEGER,
    pred_rank1_prob     REAL,
    pred_rank2_prob     REAL,
    pred_rank3_prob     REAL,
    model_version       TEXT,
    actual_rank1_car    INTEGER,
    actual_rank2_car    INTEGER,
    actual_rank3_car    INTEGER,
    rank1_hit           INTEGER,         -- 1位完全一致
    top3_hit            INTEGER,         -- 予測3車が実際のTop3に全部含まれる（任意順）
    trifecta_exact_hit  INTEGER,         -- 三連単完全一致
    settled_at          TEXT,
    FOREIGN KEY (race_id) REFERENCES races(race_id)
);

CREATE INDEX IF NOT EXISTS idx_predlog_race  ON prediction_log(race_id);
CREATE INDEX IF NOT EXISTS idx_predlog_date  ON prediction_log(predicted_at);
CREATE INDEX IF NOT EXISTS idx_predlog_model ON prediction_log(model_version);
