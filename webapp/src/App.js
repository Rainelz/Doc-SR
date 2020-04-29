import React from 'react';
import { get, post } from 'axios';
import './App.css';
import Show from './Show';
import Select from 'react-select'

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      models: [],
      model: null,
      file: null,
      src: null,
      output: null
    };
    this.onFormSubmit = this.onFormSubmit.bind(this);
    this.onFileAdded = this.onFileAdded.bind(this);
    this.onChange = this.onChange.bind(this);
    this.handleModelChange = this.handleModelChange  .bind(this);
    this.fileUpload = this.fileUpload.bind(this);
  }

  componentDidMount() {
    console.log("componentDidMount")
    get('/server/models')
      .then(res => {
        const models = Object.keys(res.data.data)
          .map(v => {
            return { value: v, label: v }
          });
        console.log("1models " + models)
        this.setState({ models: models });
      });
  }

  shouldComponentUpdate(nextProps, nextState) {
    if (this.state.output !== nextState.output) {
      return true;
    }
    if (this.state.models !== nextState.models) {
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

  handleModelChange(e) {
    this.setState({
      model: e.value
    });
  }

  onFileAdded(file) {
    this.setState((prevState) => ({
      output: this.state.file.name
    }));
  }

  fileUpload(file) {
    const url = '/server/upload/';
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', this.state.model);
    const config = {
      headers: {
        'content-type': 'multipart/form-data'
      }
    };
    return post(url, formData, config);
  }

  render() {
    const { models } = this.state;
    return (
      <div className="App">
        <header className="App-header"></header>
        <div className="">
          <label htmlFor="gan-models">Model</label>
          <Select
            id="model"
            onChange={this.handleModelChange}
            options={ models } />
        </div>
        <Show src={this.state.src} />
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
