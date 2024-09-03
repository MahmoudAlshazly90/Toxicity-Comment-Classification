Sure! Here’s the updated `README.md` file without the clone paragraph:

```markdown
# Toxicity Comment Classification

## Description

The Toxicity Comment Classification project is a deep learning-based application designed to classify comments into various categories of toxicity. Using TensorFlow and Keras, this project trains a model to predict the presence of toxic attributes in comments, such as insults, threats, or other harmful content.

The project uses an LSTM-based neural network model with bidirectional layers to capture contextual information from text data. The model is trained on a dataset of comments with labeled toxicity attributes and then evaluated for precision, recall, and accuracy.

## Features

- **Text Vectorization**: Converts raw text comments into numerical data that can be fed into a neural network.
- **Bidirectional LSTM Model**: Utilizes bidirectional Long Short-Term Memory (LSTM) layers to understand context and sequence in text.
- **Evaluation Metrics**: Computes precision, recall, and accuracy for model performance assessment.
- **Comment Scoring Interface**: Provides a web interface to score new comments and determine their toxicity.

## Dependencies

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Google Colab
- Gradio (for the web interface)

## Setup

1. **Install Dependencies**

   Ensure you have all required Python packages installed. You can use `pip` to install the dependencies:

   ```bash
   pip install tensorflow numpy pandas matplotlib gradio
   ```

2. **Mount Google Drive**

   In Google Colab, use the following code to mount Google Drive and access the dataset:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Load and Preprocess Data**

   Modify the file path in the script to point to your CSV dataset:

   ```python
   df = pd.read_csv("/content/drive/MyDrive/Toxicity comments /train.csv")
   ```

4. **Train the Model**

   Run the script to train the model. The script will save the trained model to `toxicity.h5`.

5. **Evaluate and Test**

   The script includes evaluation metrics and a sample prediction for testing the model's performance.

6. **Run the Web Interface**

   Launch the Gradio interface to score new comments:

   ```python
   interface = gr.Interface(fn=score_comment,
                            inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                            outputs='text')
   interface.launch(share=True)
   ```

## Usage

- **Training**: Modify the script parameters if necessary and execute to train the model.
- **Testing**: Use the provided test batch to evaluate model performance.
- **Prediction**: Use the `score_comment` function to analyze new comments through the Gradio web interface.

## Example

To score a new comment, use the web interface provided by Gradio. For instance, inputting "You freaking suck! I am going to hit you." will return predictions for each toxicity category.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
