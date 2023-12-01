from sanic import Sanic
from sanic.response import html, file

app = Sanic(__name__)

@app.route('/')
async def index(request):
    return await file('index.html')

connected = set()
state = ['w']*64

@app.websocket('/ws')
async def sendToOthers(request, ws):
    connected.add(ws)
    global state
    await ws.send(",".join(state))
    try:
        while True:
            message = await ws.recv()
            state = message.split(',')
            for client in connected:
                if client is not ws:
                    await client.send(message)
    finally:
        connected.remove(ws)

if __name__ == '__main__':
    app.run()

