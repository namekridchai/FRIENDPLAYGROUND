const express = require("express");
const http = require('http');
const cors = require('cors')
const app = express();
const server = http.createServer(app);
app.use(cors())
const { Server } = require("socket.io");
const io = new Server(server, {
  cors: {
    origin: "http://localhost:5173"
  }
});
const port = 3000;
const port2 =3001 ;
const playerInfos = []
let count_player = 0

svListening = app.listen(port, () => {
  console.log(`Server running at <http://localhost>:${port}/`);
});
server.listen(port2, () => {
  console.log(`Server running at <http://localhost>:${port2}/`);
});


// Middleware for logging
app.use((req, res, next) => {
  console.log(`${req.method} request for ${req.url}`);
  next();
});

// Middleware for parsing JSON
app.use(express.json());

app.post('/game', async(req, res) => {
    const state = req.body
    console.log(state)
    io.sockets.emit('new-message', state)
    console.log('emit')
    res.sendStatus(200)
  });
  // รอการ connect จาก client
  io.on('connection', client => {
      console.log(`user connected ${client.id}`)
      info  = {socket:client.id,playerSeq:count_player}
      playerInfos.push(info)
      console.log(info)
      // เมื่อ Client ตัดการเชื่อมต่อ
      client.on('disconnect', () => {
          console.log('user disconnected')
      })

      client.on('sent-message-server', function (message) {
        io.sockets.emit('new-message-server', message)
      })
  });

