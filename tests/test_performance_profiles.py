import statistics
import time
import unittest


class MockInference:
    def __init__(self, tokens: int, latency: float):
        self.tokens = tokens
        self.latency = latency

    def run(self):
        start = time.perf_counter()
        time.sleep(self.latency)
        end = time.perf_counter()
        duration = end - start
        return {
            "tokens": self.tokens,
            "duration": duration,
            "tokens_per_second": self.tokens / duration,
        }


def run_benchmark(runs, tokens, latency):
    results = []
    for _ in range(runs):
        bench = MockInference(tokens, latency)
        results.append(bench.run())
    return results


class PerformanceBenchmarkTests(unittest.TestCase):
    def test_latency_distribution_and_throughput(self):
        runs = run_benchmark(runs=3, tokens=16, latency=0.005)
        latencies = [result["duration"] for result in runs]
        throughput = [result["tokens_per_second"] for result in runs]

        self.assertTrue(all(latency > 0 for latency in latencies))
        p95 = statistics.quantiles(latencies, n=100)[94]
        self.assertLess(p95, 0.02)
        self.assertTrue(all(tp > 0 for tp in throughput))
        self.assertGreater(statistics.mean(throughput), 500)


if __name__ == "__main__":
    unittest.main()
