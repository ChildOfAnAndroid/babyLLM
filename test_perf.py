import time
import asyncio
import threading

def bench_sleep_threading(n):
    start = time.time()
    threads = []
    for _ in range(n):
        def f():
            time.sleep(0.01)
        t = threading.Thread(target=f)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    end = time.time()
    return end - start

async def bench_sleep_asyncio(n):
    start = time.time()
    async def f():
        await asyncio.sleep(0.01)

    tasks = [asyncio.create_task(f()) for _ in range(n)]
    await asyncio.gather(*tasks)

    end = time.time()
    return end - start

def main():
    n = 1000
    print("Running threading benchmark...")
    threading_time = bench_sleep_threading(n)
    print(f"Threading time for {n} concurrent sleeps: {threading_time:.4f}s")

    print("Running asyncio benchmark...")
    asyncio_time = asyncio.run(bench_sleep_asyncio(n))
    print(f"Asyncio time for {n} concurrent sleeps: {asyncio_time:.4f}s")
    print(f"Improvement: {threading_time / asyncio_time:.2f}x")

if __name__ == "__main__":
    main()
