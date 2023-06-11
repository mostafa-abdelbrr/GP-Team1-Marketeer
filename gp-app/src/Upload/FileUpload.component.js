import React, { useState } from "react";
import "./FileUpload.css";
import Logo from "../Images/Logo.png";
import Cloud from "../Images/cloud1.png";
import axios from "axios";
import LoadingSpinner from "../Loading/LoadingSpinner";
import Pie from "../Images/Pie.jpg";

function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadedCheck, setUploadedCheck] = useState(true);
  const [UploadedMsg, setUploadedMsg] = useState("");
  const [BarChart, setBarChart] = useState("");
  const [PieChart, setPieChart] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(false);
  const [zipCheck, setZipCheck] = useState(false);
  const [thirdChart, setThirdChart] = useState("");

  function checkExistOrnot(str1) {
    if (str1.indexOf(".zip") >= 0) return true;
    else return false;
  }

  // Function to handle third chart
  function handleThirdChart(data) {
    setThirdChart(data);
  }

  // Function to handle compressed file check
  function handleZipCheck(value) {
    setZipCheck(value);
  }

  // Function to handle showing results
  function handleResults(value) {
    setResults(value);
  }

  // Function to handle loading
  function handleLoading(value) {
    setIsLoading(value);
  }

  // Function to set Bar Chart
  function handleBarChart(data) {
    setBarChart(data);
  }

  // Function to set Pie Chart
  function handlePieChart(data) {
    setPieChart(data);
  }

  // Function to handle upload check
  function handleUploadedCheck(value) {
    setUploadedCheck(value);
  }

  // Function to handle Uplaod message
  function handleUploadedMsg(data) {
    setUploadedMsg(data);
  }

  // Function to handle file selection
  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  // Function to handle form submission
  const handleSubmit = async (event) => {
    // event.preventDefault();
    console.log("Uploading file...");

    // Perform upload to the backend using the selected file and fill each state with its corresponding data returning from the backend API ( UploadedMsg, BarChart, PieChart )
    // and handle loading and results. Disable the loading variable after the upload is done and enable the results variable after the upload is done.
    if (selectedFile) {
      handleLoading(true);
      handleUploadedCheck(false);
      if (checkExistOrnot(selectedFile.name)) {
        console.log("zip file");
        setZipCheck(true);
      }
      // console.log(checkExistOrnot(selectedFile.name));
      ///////////////////////// Use The Code Below /////////////////////////////////
      const instance = axios.create({ baseURL: "http://localhost:8080" });

      await instance
        .postForm("/analytics", {
          userFile: selectedFile,
        })
        .then(
          (response) => {
            console.log(response);
            handleUploadedMsg(response.data);
            handleBarChart(response.data.images[0]);
            handlePieChart(response.data.images[1]);
            if (zipCheck) {
              handleThirdChart(response.data.images[2]);
            }
            handleLoading(false);
            handleResults(true);
          },
          (error) => {
            console.log(error);
          }
        );
    }
  };

  return (
    <div>
      {uploadedCheck && (
        <div>
          <div className="header-container-wrapper">
            <div className="header-container">
              <div className="custom-header-bg">
                <div className="page-center">
                  <div className="header-columning">
                    <img src={Logo} alt="logo" className="logo-img" />
                    <div className="Logo">Marketeer</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="upload-container">
            {/* <h1 className="SText">Please click the button below to upload</h1> */}
            <img src={Cloud} alt="cloud" className="cloud-img" />
            <form className="upload-form" onSubmit={handleSubmit}>
              <input
                type="file"
                className="file-input"
                onChange={handleFileSelect}
                accept="video/*,.zip,.rar,.7zip"
                id="file"
              />
              <label htmlFor="file" className="file-input-label">
                Select File
              </label>
              {selectedFile && (
                <span className="FileName">
                  '{selectedFile.name}' is selected
                </span>
              )}
              <button type="submit" className="upload-button">
                Upload
              </button>
            </form>
          </div>
          <div className="footer-container-wrapper">
            <div className="footer-container">
              <div className="custom-footer-bg">
                <div className="page-center">
                  <p>©Marketeer</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      {isLoading && (
        <div>
          <div className="header-container-wrapper">
            <div className="header-container">
              <div className="custom-header-bg">
                <div className="page-center">
                  <div className="header-columning">
                    <img src={Logo} alt="logo" className="logo-img" />
                    <div className="Logo">Marketeer</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="upload-container">
            <p className="Spinner-Text">
              Please wait while we upload and process your video...
            </p>
            <LoadingSpinner />
          </div>
          <div className="footer-container-wrapper">
            <div className="footer-container">
              <div className="custom-footer-bg">
                <div className="page-center">
                  <p>©Marketeer</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      {results && (
        <div>
          <div className="header-container-wrapper">
            <div className="header-container">
              <div className="custom-header-bg">
                <div className="page-center">
                  <div className="header-columning">
                    <img src={Logo} alt="logo" className="logo-img" />
                    <div className="Logo">Marketeer</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="Result-container">
            <div className="box">
              <div>
                <img
                  src={`data:image/jpeg;base64,${BarChart}`}
                  alt="Bar Chart"
                  className="chart-img"
                />
                <p className="Result-Text">
                  {" "}
                  As the bar graph shows, time people pass by and inspects these
                  shelves is long which indicates that people have high interest
                  toward products on these shelves. We recommend that is you
                  want to let a specific product to be noticed by large number
                  of people to display is on these shelves.{" "}
                </p>
              </div>
              <div>
                <img
                  src={`data:image/jpeg;base64,${PieChart}`}
                  alt="Pie Chart"
                  className="chart-img"
                  responsive
                />
                <p className="Result-Text">
                  {" "}
                  As the pie chart shows, time people touch shelves is not high
                  Maybe because there are more varaieties for different products
                  on these shelves so it makes people more hesitant to buy from
                  them. We recommend that trying to display the products on more
                  area so that people be more decisive.{" "}
                </p>
              </div>
            </div>
            <div className="box">
              <div>
                <img src={Pie} alt="Pie Chart" className="chart-img" />

                <p className="Result-Text">
                  {" "}
                  As the pie chart shows, time people ispecting products is high
                  which reflects that people may have high tendancy to buy
                  products from these shelves
                </p>
              </div>
              <div>
                {zipCheck && (
                  <div>
                    <img
                      src={`data:image/jpeg;base64,${thirdChart}`}
                      alt="Pie Chart"
                      className="chart-img"
                    />

                    <p className="Result-Text">
                      {" "}
                      As the pie chart shows, time people ispecting products is
                      high which reflects that people may have high tendancy to
                      buy products from these shelves
                    </p>
                  </div>
                )}
              </div>
            </div>
            {/* <p className="Result-Text">{UploadedMsg.message}</p> */}
          </div>
          <div className="footer-container-wrapper">
            <div className="footer-container">
              <div className="custom-footer-bg">
                <div className="page-center">
                  <p>©Marketeer</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadPage;
