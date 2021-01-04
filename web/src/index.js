let express = require("express");
let app = express();
let port = 8080;

app.use(express.static("./src"));

app.listen(port, function () {
  console.log(`Listening at http://localhost:${port}`);
});