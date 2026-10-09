from sanic import Sanic
from sanic.response import file

app = Sanic(__name__)

@app.route('/')
async def index(request):
    return await file('index.html')

connected = set()
state = ['w'] * 64

@app.websocket('/ws')
async def sendToOthers(request, ws):
    global state
    connected.add(ws)
    await ws.send(",".join(state))
    try:
        while True:
            message = await ws.recv()
            cases = message.split(',')
            # Ignorer un message qui ne décrit pas une grille 8 x 8 valide
            if len(cases) != 64 or not set(cases) <= {'w', 'b'}:
                continue
            state = cases
            # list() : une copie, car connected peut changer pendant await
            for client in list(connected):
                if client is not ws:
                    await client.send(message)
    finally:
        connected.discard(ws)

if __name__ == '__main__':
    app.run()
