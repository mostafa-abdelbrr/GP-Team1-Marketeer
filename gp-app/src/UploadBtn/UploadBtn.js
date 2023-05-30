// import FileUpload from "./Upload/FileUpload.component";
import React, { useState } from "react";
import "./UploadBtn.css";
// import FileUpload from "./components/file-upload/file-upload.component";

function UploadBtn() {
  const [newUserInfo, setNewUserInfo] = useState({
    profileImages: [],
  });

  const updateUploadedFiles = (files) =>
    setNewUserInfo({ ...newUserInfo, profileImages: files });

  const handleSubmit = (event) => {
    event.preventDefault();
    //logic to create new user...
  };

  return (
    <div class="container">
      <div class="vertical-center">
        <button type="submit" className="next-button">
        Next
      </button>
      </div>
    </div>
  );
}

export default UploadBtn;
