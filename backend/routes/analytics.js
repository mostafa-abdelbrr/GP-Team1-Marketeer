var express = require("express");
var router = express.Router();
const multer = require("multer");
const path = require("path");
const upload = multer({
    dest: "uploads/"
});
const spawn = require("child_process").spawn;
const zip = require("node-stream-zip");
const fs = require("fs");

router.get("/", (req, res) => {
    // res.json({message: "Analytics endpoint is up."});
    const fileName = "/screenshots/frame0.jpg";
    res.json({
        dataAnalysis: path.resolve(__dirname, "../../analytics/" + fileName)
    });
});

router.post("/", upload.single("userFile"), (req, res) => {
    console.log("Prcessing file:");
    console.log(req.file);
    var pyProg;
    var msg = "";
    if (req.file.originalname.split(".").pop() == "zip") {
        
        const StreamZip = require('node-stream-zip');
        const zip = new StreamZip({ file: req.file.path });
        zip.on('ready', () => {
            // fs.mkdirSync('extracted');
            fs.mkdirSync('extracted/' + req.file.filename);
            zip.extract(null, "extracted/" + req.file.filename + "/" , (err, count) => {
                console.log(err ? 'Extract error' : `Extracted ${count} entries`);
                zip.close();
            });
        });
        // Handle errors
        zip.on('error', err => { /*...*/ });
        pyProg = spawn("python", [
            "../analytics/analytics.py", "\\extracted\\" + req.file.filename
        ]);
    } else {
        pyProg = spawn("python", ["../analytics/analytics.py", "\\uploads\\" + req.file.filename]);
        // msg = req.file.originalname.split(".").pop() + "\n";
    }
    pyProg.stdout.on("data", function(data) {
        console.log(data.toString());
        msg += data.toString() + "\n";
    });
    pyProg.stderr.on("data", function(data) {
        console.log(data.toString());
    });
    pyProg.on("exit", function(code, signal) {
        var plots = [];
        var filenames = fs.readdirSync('analytics/');
        for (const filename of filenames) {
            var encodedPlot = fs.readFileSync('analytics/' + filename, {encoding: 'base64'});
            plots.push(encodedPlot.toString());
        }
        plots.push('')
        res.json({
            shelve_interaction: {
                label: msg.split('\r\n')[0],
                plot: plots[0]
            },
            shelve_attention_time:{
                label: msg.split('\r\n')[1],
                plot: plots[1]
            },
            shelve_inspection: {
                label: msg.split('\r\n')[2],
                plot: plots[2]
            },
            product_frequency: {
                label: 'The bar plot shows the frequency of people inspecting each product.',
                plot: plots[3]
            }
        });

        console.log("Sent response.");
        fs.unlink(req.file.path, err => {
            if (err) {
                console.error("Failed to delete the file:", err);
            } else {
                console.log("File deleted successfully");
            }
        });
        if (req.file.originalname.split(".").pop() == "zip") {
            fs.rm("extracted\\" + req.file.filename, { recursive: true }, (err) => {
                if (err) {
                console.error('Failed to delete the folder:', err);
                } else {
                console.log('Folder deleted successfully');
                }
            });
        }
        var filenames = fs.readdirSync('analytics/');
        for (const filename of filenames) {
            fs.rmSync("analytics/" + filename);
        }
    });
});
module.exports = router;
