var express = require("express");
var router = express.Router();
const multer  = require('multer');
const path = require('path');
const upload = multer({ dest: 'uploads/' });
const spawn = require('child_process').spawn;

router.get('/', (req, res, ) => {
    // res.json({message: "Analytics endpoint is up."});
    const fileName = '/screenshots/frame0.jpg';
        const options = {
            root: path.resolve(__dirname, '../../analytics/')
        };
        res.sendFile(fileName, options, function (err) {
            if (err) {
                console.log(path.resolve(__dirname, '../../analytics/'));
                console.log(err);
            } else {
                console.log('Sent:', fileName);
            }
        });
    
});

router.post('/', upload.single('file'), (req, res) => {
    console.log('Prcessing file:');
    console.log(req.file);
    const pyProg = spawn('python', ['../analytics/analytics.py', '\\uploads\\' + req.file.filename]);

    pyProg.stdout.on('data', function(data) {
        console.log(data.toString());
        res.write(data);
    });

    pyProg.on('exit', function(code, signal) {
        // const fileName = '/screenshots/frame0.jpg';
        // const options = {
        //     root: path.resolve(__dirname, '../../analytics/')
        // };
        // res.sendFile(fileName, options, function (err) {
        //     if (err) {
        //         next(err);
        //     } else {
        //         console.log('Sent:', fileName);
        //     }
        // });
        res.end();
        console.log('Sent response.')
    });
});
module.exports = router