// import React, { useState } from "react";
// import "../Sequence/MainSequence.css";
// import LandingPage from "../LandingPage/LandingPage";
// import FileUpload from "../Upload/FileUpload.component";

// const MainSequence = () => {
//     const [nextClicked, setNextClicked] = useState(false);

//     const handleClick = (click) => {
//       setNextClicked(click);
//     };

//     const [uploadClicked, setUploadClicked] = useState(false);

//     const handleUploadClick = (click) => {
//         setUploadClicked(click);
//     };

//     const [newUserInfo, setNewUserInfo] = useState({
//       profileImages: [],
//     });

//     const updateUploadedFiles = (files) =>
//       setNewUserInfo({ ...newUserInfo, profileImages: files });
    
//     state = {
//       step:1,
//     }

//     return (
//         <div>
//             {!nextClicked && <LandingPage handleClick={handleClick} />}
//             {nextClicked && !uploadClicked && <FileUpload
//           accept="video/*"
//           label="Videos to process"
//           multiple
//           updateFilesCb={updateUploadedFiles}
//         />}

//         </div>
//     );
// }

// export default MainSequence;

import React, { Component, useState } from "react";
import "../Sequence/MainSequence.css";
import LandingPage from "../LandingPage/LandingPage";
import FileUpload from "../Upload/FileUpload.component";
import Results from "../Results/Results";

export default class MainSequence extends Component {
  
  state = {
    step: 1,
  };
  // proceed to the next step
  nextStep = () => {
    const { step } = this.state;
    this.setState({ step: step + 1 });
  };
  // handle field change
  handleChange = (input) => (e) => {
    this.setState({ [input]: e.target.value });
  };

  render() {

    const { step } = this.state;
    switch (step) {
      case 1:
        return(
        <LandingPage nextStep={this.nextStep} handleChange={this.handleChange} />
        );
      case 2:
        return (
          <FileUpload
            nextStep={this.nextStep}
            handleChange={this.handleChange}
            updateFilesCb={this.props.updateUploadedFiles}
          />
        );
      case 3:
        return (
          <Results/>
        );
      default:
      // return ();
      // <div>
      //   hello
      //   {/* <LandingPage/> */}
      // </div>
    }
  }
}