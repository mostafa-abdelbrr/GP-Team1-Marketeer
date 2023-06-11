import React, { Component } from "react";
import "./LandingPage.css";
import Logo from "../Images/Logo.png";
import Sample from "../Images/Sample.png";

export class LandingPage extends Component {
  Continue = (e) => {
    e.preventDefault();
    this.props.nextStep();
  };
  render() {
    return (
      <div className="Title">
        <div>
          <div className="Title">
            <img src={Logo} alt="logo" className="logo" />
            <div className="title"> Marketeer </div>
          </div>
          <p className="Vision"> Our Vision </p>
          <p className="VisionText">
            {" "}
            Market owners have always sought practical ways to effectively
            understand how their customers behave in order to provide an easy
            and delightful shopping experience that draws more customers and
            boosts sales. Humans are naturally adept at analysing behaviour from
            a variety of senses, including actions, gestures, and other visuals.
            But making computer vision systems capable of doing the same is a
            challenging goal. Online stores find it easy to track customer
            shopping behaviour, but it is more challenging for offline stores to
            do the same. There are inherent difficulties involved, along with
            external obstacles like limited access to relevant data and varying
            environmental conditions.{" "}
          </p>
          <p className="VisionText2">
            {" "}
            In our project, we focus on helping stores collect statistics about
            how customers behave by analysing the customer behaviour and
            interaction with the products which are showed through our software
            application interface, that is, by using various computer vision
            techniques.{" "}
          </p>
          <button className="NextBt" onClick={this.Continue}>
            {" "}
            Next{" "}
          </button>
        </div>
        <img src={Sample} alt="logo" className="sample" />
      </div>
    );
  }
}

export default LandingPage;