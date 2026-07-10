import asyncio


async def count():
    await asyncio.sleep(5)
    print("we counted to 5")


async def try_counting():
    print("before starting the count")
    result = await count()
    print("after count")
    print(f"async result: {result}")


asyncio.run(try_counting())
