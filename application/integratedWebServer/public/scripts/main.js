var rhit = rhit || {};
rhit.stockUrl = 'http://localhost:3000/api/'
// rhit.stockUrl = 'https://stock-predictions-673a4.web.app'
rhit.tickName = class {
    constructor(tick) {
        this.ticker = tick;
    }
    readFile() {
        console.log(`TODO: Read all the words from the backend, then update the screen.`);
		
		
        // TODO: Add your code here.
        var tick = this.ticker;
        // console.log('surl :>> ', surl);
        console.log('`${rhit.stockUrl}${this.ticker}` :>> ', `${rhit.stockUrl}${tick}`);
        // module.exports = {tick};
		var temp = {'ticker':tick};
        let allEntries = fetch(`${rhit.stockUrl}${tick}`,
        {
            method: "PUT",
            headers: { "Content-Type": 'application/json' },
            body: JSON.stringify(temp)
        })
        .then(response => response.json())
        .then(data => {
			console.log('data :>> ', data);
			console.log('data[0] :>> ', data['data']);
			document.querySelector("#scripted").innerHTML = data['data'][1][0];
            console.log('data in main :>> ', data);
           
        })
        // Hint for something you will need later in the process (after backend call(s))
    }


    // readData = function () {
    //  const spawner = require('child_process').spawn;
    //  const data_to_pass_in = this.ticker;
    //  console.log('data_to_pass_in :>> ', data_to_pass_in);
    //  const python_process = spawner('python', ['./python.py', JSON.stringify(data_to_pass_in)]);
    //  python_process.stdout.on('data', (data) => {
    //      console.log('Data received: ', JSON.parse(data.toString()));
    //  });
    //  return JSON.parse(data.toString());
    //  // var data = $.csv.toObjects(csv);
    //  // console.log('data :>> ', data);


    // }
}


rhit.submitController = class {
    constructor() {
        document.querySelector('#submit').onclick = (event) => {
           
            const ticker = document.querySelector('#symbol').value;
            var tickObj = new rhit.tickName(ticker);
			
            document.querySelector('#temp').innerHTML = `You entered: ${ticker}`;
            localStorage.setItem("ticker", tickObj.ticker);
            document.querySelector('#scripted').innerHTML = `File to read is: ${tickObj.readFile()}`;
            // window.location.href = `${rhit.stockUrl}/${tickObj.ticker}`;
            // console.log('ticker :>> ', ticker);
            // this._createHTML(ticker);
        }
    }


   




    // htmlToElement(html) {
    //  var template = document.createElement("template");
    //  html = html.trim();
    //  template.innerHTML = html;
    //  return template.content.firstChild;
    // }


    // _createHTML(ticker){
    //  console.log('ticker :>> ', ticker);
    //  return this.htmlToElement(`
    //  <div id="temp">You entered: ${ticker}</div>


    //  `)
    // }
}
/* Main */
/** function and class syntax examples */
rhit.main = function () {
    rhit.controller = new rhit.submitController();
};


rhit.main();



