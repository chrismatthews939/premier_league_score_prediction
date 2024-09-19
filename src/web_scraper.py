import requests
from bs4 import BeautifulSoup
import os

# Get the absolute path to the 'data' directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(project_root, 'data')

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# n=3 is the number of files to extract from the most recent. Change this to get more history
def download_latest_premier_league_csvs(n=3):
    """
    This function downloads the latest n Premier League CSV files from football-data.co.uk.
    """
    # Step 1: Send a GET request to the website
    response = requests.get('https://www.football-data.co.uk/englandm.php')
    
    # Check if the request was successful (status code 200)
    if response.status_code != 200:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return
    
    # Step 2: Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Step 3: Find all links to Premier League CSV files (files are called E0.csv on site) on the page
    csv_links = soup.find_all('a', href=True)
    premier_league_links = [link for link in csv_links if 'E0.csv' in link['href']]
    
    # Step 4: Select the most recent 'n' links (latest seasons are listed first)
    latest_links = premier_league_links[:n]
    
    # Step 5: Download each of the selected CSV files
    for link in latest_links:
        # Construct the full URL of the CSV file
        csv_url = 'https://www.football-data.co.uk/' + link['href']
        
        # Get the name of the CSV file from the URL
        file_name = link['href'].split('/')[-1]
        
        # Path where the file will be saved
        file_path = os.path.join(output_dir, file_name)
        
        # Step 6: Send a GET request to download the CSV
        print(f"Downloading {file_name}...")
        csv_response = requests.get(csv_url)
        
        # Step 7: Save the CSV content to a file
        with open(file_path, 'wb') as file:
            file.write(csv_response.content)
        print(f"{file_name} saved to {output_dir}")
    
    print(f"Saving file to {file_path}")
    print(f"The latest {n} Premier League CSV files downloaded successfully!")

if __name__ == "__main__":
    # Call the function to download the latest 3 CSVs
    download_latest_premier_league_csvs(n=3)
