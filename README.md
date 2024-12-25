# News Article Retrieval System
### Project by: Dekkar Raid / Ouaret Zineddine

## Overview
The News Article Retrieval System is a search engine designed to process and search through a dataset of BBC news articles. The system enables users to search for news articles by keywords and filter results based on specific categories. Additionally, users can visualize the dataset's directory structure for better comprehension.

## Features
- **Search Functionality**: Retrieve articles containing specific keywords.
- **Category Filtering**: Filter search results by predefined categories.
- **Article Display**: View detailed content of individual articles.
- **Dataset Visualization**: Visualize the structure of the dataset directory.

## Dataset
The system uses the "bbc-news-data.csv" dataset, which contains news articles categorized by their topics such as business, sports, technology, etc. Each row in the dataset includes:
- `category`: The topic of the article.
- `filename`: The file identifier for the article.
- `title`: The headline of the article.
- `content`: The full text of the article.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/News-Article-Retrieval-System.git
   cd News-Article-Retrieval-System
   ```

2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset (`bbc-news-data.csv`) in the dataset directory.

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your web browser and go to `http://127.0.0.1:5000`.

## File Structure
```
project-folder/
├── app.py               # Main application script
├── templates/           # HTML templates for the web interface
│   ├── index.html       # Home page
│   ├── search.html      # Search results page
│   └── visualization.html     # Article display page
├── static/
│   ├── css/
│   │   └── main.css    # CSS styles
│   └── js/             # Optional JavaScript files (if needed)
├── dataset/
│   └── bbc-news-data.csv    # Dataset file
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## How It Works

1. **Loading the Dataset**:
   The application loads the `bbc-news-data.csv` file, processes its contents, and prepares it for searching.

2. **Search and Filter**:
   Users can enter keywords in the search bar on the homepage. Optionally, they can select a category to narrow down the results.

3. **View Articles**:
   Each search result links to a detailed view of the article, displaying its full content and metadata.

4. **Dataset Visualization**:
   A separate feature visualizes the dataset's directory structure for better understanding.

## Contributions
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
- The BBC for providing the dataset.
- Flask, Pandas, and other libraries for enabling rapid development.
- Open-source community for inspiration and support.