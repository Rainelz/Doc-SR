import React from 'react';
import axios, { post } from 'axios';
import Show from './Show';

class App extends React.Component {
  // static getDerivedStateFromProps(props, state) {
  //   console.log(props);
  //   console.log(state);
  // }
  constructor(props) {
    super(props);
    this.state = {
      file: null,
      src: null,
      output: null
    };
    this.onFormSubmit = this.onFormSubmit.bind(this);
    this.onFileAdded = this.onFileAdded.bind(this);
    this.onChange = this.onChange.bind(this);
    this.fileUpload = this.fileUpload.bind(this);
  }

  shouldComponentUpdate(nextProps, nextState) {
    if (this.state.output !== nextState.output) {
      return true;
    }
    return false;
  }

  onFormSubmit(e) {
    e.preventDefault(); // Stop form submit
    this.fileUpload(this.state.file).then((response) => {
      this.setState((prevState) => ({
        file: this.state.file,
        src: this.state.file.name,
        output: this.state.file.name
      }));
    });
  }

  onChange(e) {
    this.setState({
      file: e.target.files[0]
    });
  }

  onFileAdded(file) {
    this.setState((prevState) => ({
      output: this.state.file.name
    }));
  }

  fileUpload(file) {
    const url = '/api/upload';
    const formData = new FormData();
    formData.append('file', file);
    const config = {
      headers: {
        'content-type': 'multipart/form-data'
      }
    };
    return post(url, formData, config);
  }

  render() {
    return (
      <div>
        <Show src={this.state.src} output={this.state.output} />
        <form onSubmit={this.onFormSubmit}>
          <h1>File Upload</h1>
          <input type="file" onChange={this.onChange} />
          <button type="submit">Upload</button>
        </form>
      </div>
    );
  }
}

export default App;
