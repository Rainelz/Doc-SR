import React from 'react';
import BeforeAfterSlider from 'react-before-after-slider';

class Show extends React.Component {
  render() {
    if (this.props.src) {
      const file1 = 'http://localhost:5000/uploads/' + this.props.src;
      const file2 = 'http://localhost:5000/uploads/' + this.props.src.replace('.png', '_processed.png');
      return (
        <BeforeAfterSlider
          before={file1}
          after={file2}
          width={640}
          height={480}
        />
      );
    } else {
      return <div>oh no</div>;
    }
  }
}

export default Show;
