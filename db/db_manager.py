import sqlite3, os, uuid
import multiprocessing as mp


class DBManager:

    def __init__(self, db_name="kernel_turbo.db", db_path="db"):
        os.makedirs(db_path, exist_ok=True)
        self.db_path = os.path.join(db_path, db_name)
        self.queue = mp.Queue()
        self.process = mp.Process(target=self._db_process_loop,
                                  args=(self.queue, ))
        self.process.start()

    def _db_process_loop(self, queue):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        self._create_tables(cursor)
        conn.commit()

        while True:
            task = queue.get()
            if task is None:
                break
            try:
                func_name, args = task
                getattr(self, f"_do_{func_name}")(cursor, *args)
                conn.commit()
            except Exception as e:
                print("[DB] Error:", e)

        conn.close()

    # ---------------- 表创建 ----------------
    def _create_tables(self, cursor):
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            provider TEXT NOT NULL,
            temperature REAL DEFAULT 0.0,
            max_tokens INTEGER DEFAULT 4096,
            top_p REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, provider)
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id TEXT PRIMARY KEY,
            description TEXT,
            system_prompt TEXT NOT NULL,
            prompt TEXT NOT NULL,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id TEXT PRIMARY KEY,
            iteration INTEGER DEFAULT 1,
            raw_code TEXT,
            gen_code TEXT,
            iter_prompt TEXT,
            model_id TEXT NOT NULL,
            prompt_id TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_id) REFERENCES models(id),
            FOREIGN KEY (prompt_id) REFERENCES prompts(id),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )""")

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS benchmarks (
            id TEXT PRIMARY KEY,
            generation_id TEXT NOT NULL,
            compile_message TEXT,
            validate_message TEXT,
            raw_code_exec_time REAL,
            gen_code_exec_time REAL,
            speedup REAL,
            status INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (generation_id) REFERENCES generations(id)
        )""")

    # ---------------- 插入接口 ----------------
    def insert_experiment(self, name):
        exp_id = str(uuid.uuid4())
        self.queue.put(("insert_experiment", (exp_id, name)))
        return exp_id

    def insert_model(self,
                     name,
                     provider,
                     temperature=0.0,
                     max_tokens=4096,
                     top_p=1.0):
        model_id = str(uuid.uuid4())
        self.queue.put(("insert_model", (model_id, name, provider, temperature,
                                         max_tokens, top_p)))
        return model_id

    def insert_prompt(self, system_prompt, prompt, description=None):
        prompt_id = str(uuid.uuid4())
        self.queue.put(
            ("insert_prompt", (prompt_id, system_prompt, prompt, description)))
        return prompt_id

    def insert_generation(self, iteration, raw_code, gen_code, iter_prompt,
                          model_id, prompt_id, experiment_id):
        gen_id = str(uuid.uuid4())
        self.queue.put(("insert_generation",
                        (gen_id, iteration, raw_code, gen_code, iter_prompt,
                         model_id, prompt_id, experiment_id)))
        return gen_id

    def insert_benchmark(self,
                         generation_id,
                         compile_message=None,
                         validate_message=None,
                         raw_code_exec_time=None,
                         gen_code_exec_time=None,
                         speedup=None,
                         status=-1):
        bench_id = str(uuid.uuid4())
        self.queue.put(
            ("insert_benchmark",
             (bench_id, generation_id, compile_message, validate_message,
              raw_code_exec_time, gen_code_exec_time, speedup, status)))
        return bench_id

    # ---------------- 执行函数 ----------------
    def _do_insert_experiment(self, cursor, id, name):
        cursor.execute(
            "INSERT OR IGNORE INTO experiments (id, name) VALUES (?, ?)",
            (id, name))
        print(f"[DB] Inserted experiment: {name} (id={id})")

    def _do_insert_model(self, cursor, id, name, provider, temperature,
                         max_tokens, top_p):
        cursor.execute(
            """INSERT OR IGNORE INTO models (id, name, provider, temperature, max_tokens, top_p)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (id, name, provider, temperature, max_tokens, top_p))
        print(f"[DB] Inserted model: {name} ({provider}), id={id}")

    def _do_insert_prompt(self, cursor, id, system_prompt, prompt,
                          description):
        cursor.execute(
            """INSERT INTO prompts (id, system_prompt, prompt, description)
               VALUES (?, ?, ?, ?)""",
            (id, system_prompt, prompt, description))
        print(f"[DB] Inserted prompt: id={id}")

    def _do_insert_generation(self, cursor, id, iteration, raw_code, gen_code,
                              iter_prompt, model_id, prompt_id, experiment_id):
        cursor.execute(
            """INSERT INTO generations (id, iteration, raw_code, gen_code, iter_prompt, model_id, prompt_id, experiment_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (id, iteration, raw_code, gen_code, iter_prompt, model_id,
             prompt_id, experiment_id))
        print(
            f"[DB] Inserted generation {id} for experiment_id {experiment_id}")

    def _do_insert_benchmark(self, cursor, id, generation_id, compile_message,
                             validate_message, raw_code_exec_time,
                             gen_code_exec_time, speedup, status):
        cursor.execute(
            """INSERT INTO benchmarks (id, generation_id, compile_message, validate_message,
                                       raw_code_exec_time, gen_code_exec_time, speedup, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (id, generation_id, compile_message, validate_message,
             raw_code_exec_time, gen_code_exec_time, speedup, status))
        print(
            f"[DB] Inserted benchmark {id} for generation_id {generation_id}")

    # ---------------- 关闭 ----------------
    def shutdown(self):
        self.queue.put(None)
        self.process.join()
