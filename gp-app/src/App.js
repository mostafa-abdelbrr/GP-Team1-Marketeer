import React, { useState } from "react";
import "./App.css";
import "./LandingPage/LandingPage";
// import FileUpload from "./components/file-upload/file-upload.component";
import MainSequence from "./Sequence/MainSequence";
import "./Fonts/SamsungSans-Regular.ttf";
import "./Fonts/SamsungSans-Light.ttf";
import "./Fonts/SamsungSans-Bold.ttf";
import "./Fonts/SamsungSans-Medium.ttf";
import "./Fonts/SamsungSans-Thin.ttf";
function App() {
  const [newUserInfo, setNewUserInfo] = useState({
    profileImages: [],
  });

  const updateUploadedFiles = (files) =>
    setNewUserInfo({ ...newUserInfo, profileImages: files });

  return (
    <div>
      <MainSequence updateFilesCb={updateUploadedFiles}/>
    </div>
  );
}

export default App;
