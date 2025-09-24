import requests

# 1️⃣ Plan
plan_resp = requests.post("http://127.0.0.1:8000/plan", json={"target_url":"https://example.com","n_candidates":3})
candidates = plan_resp.json()["candidates"]
print("Candidates:", candidates)

# 2️⃣ Rank
rank_resp = requests.post("http://127.0.0.1:8000/rank", json={"candidates":candidates, "top_k":2})
top_candidates = rank_resp.json()["top_candidates"]
print("Top Candidates:", top_candidates)

# 3️⃣ Execute
execute_resp = requests.post("http://127.0.0.1:8000/execute", json={"tests":top_candidates, "parallelism":2})
execute_data = execute_resp.json()
run_id = execute_data["run_id"]
results = execute_data["results"]
print("Run ID:", run_id)
print("Results:", results)

# 4️⃣ Analyze
analyze_resp = requests.post("http://127.0.0.1:8000/analyze", json={"run_id":run_id, "results":results})
report = analyze_resp.json()
print("Report Path:", report["report_path"])
print("Report:", report["report"])
