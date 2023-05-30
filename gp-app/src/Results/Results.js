import React, { Component } from "react";
import axios from "axios";
export class Results extends Component {
  Continue = (e) => {
    e.preventDefault();
    console.log("Continue");
    this.props.nextStep();
  };
  onClick = () => {
    /*
    Begin by setting loading = true, and use the callback function
    of setState() to make the ajax request. Set loading = false after
    the request completes.
  */
    this.setState({ loading: true }, () => {
      axios.get("/endpoint").then((result) =>
        this.setState({
          loading: false,
          data: [...result.data],
        })
      );
    });
  };
  render() {
    const { data, loading } = this.state;

    return (
      <div>
        <button onClick={this.onClick}>Load Data</button>
        Test

        {/*
            Check the status of the 'loading' variable. If true, then display
            the loading spinner. Otherwise, display the results.
          */}
        {/* {loading ? <LoadingSpinner /> : <ResultsTable results={data} />} */}
      </div>
    );
  }
}

export default Results;