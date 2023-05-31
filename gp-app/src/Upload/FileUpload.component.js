// import React, { useRef, useState } from "react";
// import "./FileUpload.css";
// // import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
// // import { faAlternateTrash } from "@fortawesome/fontawesome-free-solid";
// import UploadBtn from "../UploadBtn/UploadBtn";
// const KILO_BYTES_PER_BYTE = 1000.0;
// const DEFAULT_MAX_FILE_SIZE_IN_BYTES = 999999999999999999999999;

// const convertNestedObjectToArray = (nestedObj) =>
//   Object.keys(nestedObj).map((key) => nestedObj[key]);

// const convertBytesToKB = (bytes) =>
//   Math.round(bytes / KILO_BYTES_PER_BYTE);

// const FileUpload = ({
//   label,
//   updateFilesCb,
//   maxFileSizeInBytes = DEFAULT_MAX_FILE_SIZE_IN_BYTES,
//   ...otherProps
// }) => {
//   const fileInputField = useRef(null);
//   const [files, setFiles] = useState({});

//   const handleUploadBtnClick = () => {
//     fileInputField.current.click();
//   };

//   const addNewFiles = (newFiles) => {
//     for (let file of newFiles) {
//       if (file.size < maxFileSizeInBytes) {
//         if (!otherProps.multiple) {
//           return { file };
//         }
//         files[file.name] = file;
//       }
//     }
//     return { ...files };
//   };

//   const callUpdateFilesCb = (files) => {
//     const filesAsArray = convertNestedObjectToArray(files);
//     updateFilesCb(filesAsArray);
//   };

//   const handleNewFileUpload = (e) => {
//     const { files: newFiles } = e.target;
//     if (newFiles.length) {
//       let updatedFiles = addNewFiles(newFiles);
//       setFiles(updatedFiles);
//       // callUpdateFilesCb(updatedFiles);
//     }
//   };

//   const removeFile = (fileName) => {
//     delete files[fileName];
//     setFiles({ ...files });
//     // callUpdateFilesCb({ ...files });
//   };

//   return (
//     <div className="Body">
//       <div className="FileUploadContainer">
//         <div className="InputLabel">   ­­</div>
//         <div className="DragDropText">Please click on the button below to upload</div>
//         <div
//           className="UploadFileBtn"
//           type="button"
//           onClick={handleUploadBtnClick}
//         >
//           <i className="UploadSign">↑</i>
//           <span> Upload {otherProps.multiple ? "files" : "a file"}</span>
//         </div>
//         <input
//           className="FormField"
//           type="file"
//           ref={fileInputField}
//           onChange={handleNewFileUpload}
//           title=""
//           value=""
//           accept="video/*"
//           {...otherProps}
//         />
//       </div>
//       <hr
//         style={{
//           height: "0.001px",
//           width: "50%",
//         }}
//       />
//       <div className="FilePreviewContainer">
//         ­
//         <div className="toupload">To Upload</div>
//         <div className="PreviewList">
//           {Object.keys(files).map((fileName, index) => {
//             let file = files[fileName];
//             let isImageFile = file.type.split("/")[0] === "image";
//             return (
//               <div className="PreviewContainer" key={fileName}>
//                 <div>
//                   {isImageFile && (
//                     <div
//                       className="ImagePreview"
//                       src={URL.createObjectURL(file)}
//                       alt={`file preview ${index}`}
//                     />
//                   )}
//                   <div className="FileMetaData" isImageFile={isImageFile}>
//                     {file.name}
//                     <aside>
//                       {convertBytesToKB(file.size)} kB
//                       <div
//                         className="RemoveFileIcon"
//                         onClick={() => removeFile(fileName)}
//                       >
//                         x
//                       </div>
//                     </aside>
//                   </div>
//                 </div>
//               </div>
//             );
//           })}
//         </div>
//       </div>
//       ­
//       <UploadBtn/>
//     </div>
//   );
// };

// export default FileUpload;

import React, { useState } from "react";
import "./FileUpload.css";
import Logo from "../Images/UploadLogoy.png";
import Cloud from "../Images/cloud1.png";
import axios, { isCancel, AxiosError } from "axios";

function UploadPage() {
  const [selectedFile, setSelectedFile] = useState(null);

  function Continue(e) {
    // e.preventDefault();
    console.log("Continue");
    this.props.nextStep();
  }

  // Function to handle file selection
  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  // Function to handle form submission
  const handleSubmit = (event) => {
    event.preventDefault();

    // Perform upload to the backend using the selected file
    if (selectedFile) {
      console.log("Uploading file...");
      // Continue();
      // Send the selectedFile to the backend
      // You can use libraries like Axios or fetch for making the HTTP request
      // Example: axios.post('/upload', selectedFile);
      // axios
      // .post("http://localhost:5000/upload", { userfiles: selectedFile }); // Send the file to the backend
      // .then((res) => { // Handle the response from the backend
      //   console.log(res); // If backend sends back a success message, update the UI to show success
      // })

      // // If you are using Axios, you can use the code below to listen to the upload progress
      // .uploadProgress((progressEvent) => {
      //   const { loaded, total } = progressEvent;
      //   const percent = Math.floor((loaded * 100) / total);
      //   console.log(`${loaded}kb of ${total}kb | ${percent}%`);
      // });
      
      ///////////////////////// Use The Code Below /////////////////////////////////
      const instance = axios.create({ baseURL: "http://localhost:8080" });

      instance
        .postForm("/analytics", {
          userFile: selectedFile,
        })
        .then(
          (response) => {
            console.log(response.data.message);
            // Continue();
          },
          (error) => {
            console.log(error);
          }
        );
    }
  };

  return (
    <div>
      <div className="header-container-wrapper">
        <div className="header-container">
          <div className="custom-header-bg">
            <div className="page-center">
              <div className="header-columning">
                <img src={Logo} alt="logo" className="logo-img" />
                <div className="Logo">Markteer</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="upload-container">
        <h1>Please click the button below to upload</h1>
        <img src={Cloud} alt="cloud" className="cloud-img" />
        <form className="upload-form" onSubmit={handleSubmit}>
          <input
            type="file"
            className="file-input"
            onChange={handleFileSelect}
            accept="video/*"
            id="file"
          />
          <label htmlFor="file" className="file-input-label">
            Select File
          </label>
          {selectedFile && <span>{selectedFile.name}</span>}
          <button type="submit" className="upload-button">
            Upload
          </button>
        </form>
      </div>
      <div className="footer-container-wrapper">
        <div className="footer-container">
          <div className="custom-footer-bg">
            <div className="page-center">
              <p>Markteer</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default UploadPage;
