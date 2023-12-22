import aiohttp
import asyncio

async def charge(url):
  print("charge ",url)
  async with aiohttp.ClientSession() as session:
    async with session.get(url) as response:
        r = await response.text()
  print("chargement de l'URL ",url," terminé")
  return r

async def main():
    task1 = asyncio.create_task(
        charge("https://www.google.com/"))

    task2 = asyncio.create_task(
        charge("https://www.uqam.ca/"))
    await asyncio.gather(task1, task2)
    print(len(task1.result()))
    print(len(task2.result()))

if __name__ == '__main__':
   asyncio.run(main())
 