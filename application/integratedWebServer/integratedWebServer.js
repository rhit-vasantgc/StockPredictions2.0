var express = require("express");
const bodyParser = require("body-parser");
const spawner = require('child_process').spawn;
const fs = require("fs");


// Setup
const app = express();
console.log("created express above");
app.use('/', express.static("public"));
app.use('/api/', bodyParser.urlencoded({
    extended: true
}));
app.use('/api/', bodyParser.json());
console.log("created api uses above");
let data = [];
const serverSideStorage = "C://Users//vasantgc//Documents//StockPredictions2.0//application//db.json";


// function readTick(){
    fs.readFile(serverSideStorage, function (err, buf) {
        if (err) {
            console.log("error: ", err);
        } else {
            data = JSON.parse(buf.toString());
            console.log('data lin 24:>> ', data);
        }
        console.log("Data read from file.");
    });
    

//     return data;
// }



function saveToServer(data) {
    fs.writeFile(serverSideStorage, JSON.stringify(data), function (err, buf) {
        if (err) {
            console.log("error: ", err);
        } else {
            console.log("Data saved successfully!");
        }
    })
}


app.put("/api/:tick",function(req,res){
    let tick = (req.params.tick);
    console.log('tick :>> ', tick);
    data[0] = tick;
   
    saveToServer(data);
    let data_to_pass_in = data;
console.log('data_to_pass_in :>> ', data_to_pass_in);
const python_process = spawner('python',['C://Users//vasantgc//Documents//StockPredictions2.0//application//integratedWebServer//public//scripts//python.py',JSON.stringify(data_to_pass_in)]);
python_process.stdout.on('data',(data) =>{
    res.send({"data":JSON.parse(data.toString())});
    res.status();
    res.end();
    console.log('data ooga:>> ', data);
    console.log('Data received: ', JSON.parse(data.toString()));
});
    // readTick();
    // let result = data[id];
    // res.send({"tick":tick});
    
})

// let a = readTick();
// console.log('a :>> ', a);
console.log('data :>> ', data);
console.log('data booga:>> ', data);


// app.get('/api/testing',function(req,res){
//     temp = {"word":data,"length":data.length}
//     res.send(temp);
//     res.status();
//     res.end();
// })





// saveToServer(['aapl']);

// console.log('localStorage.getItem("ticker") :>> ', window.localStorage.getItem("ticker"));
// console.log('surl :>> ', surl);



app.listen(3000);