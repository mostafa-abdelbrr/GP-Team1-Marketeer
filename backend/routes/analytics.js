var express = require("express");
var router = express.Router();
const multer  = require('multer')
const upload = multer({ dest: 'uploads/' })
const spawn = require('child_process').spawn;

router.get('/', (req, res, ) => {
    res.json({message: "Analytics endpoint is up."});
    
});

router.post('/', upload.single('file'), (req, res) => {
    const title = req.body.title;
    const file = req.file;
    console.log('Prcessing file:');
    console.log(req.file);
    const pyProg = spawn('python', ['../analytics/analytics.py', '\\uploads\\' + req.file.filename]);

    pyProg.stdout.on('data', function(data) {
        console.log(data.toString());
        res.write(data);
    });

    pyProg.on('exit', function(code, signal) {
        res.end();
        console.log('Sent response.')
    })
});
module.exports = router