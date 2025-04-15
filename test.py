import requests  # Make sure this is included

def get_jsonparsed_data(url):
    """
    Fetch JSON data from a URL and parse it
    """
    try:
        print(f"Attempting to fetch data from: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

def get_earnings_call(symbol, year, quarter):
    url = f"https://discountingcashflows.com/api/transcript/?ticker={symbol}&quarter={quarter}&year={year}&key=488e92b0-6a83-46a5-a81d-2b7dd21437ce"
    data = get_jsonparsed_data(url)
    if isinstance(data, list) and data:
        print("Successfully retrieved data")
        return data
    else:
        print(f"Error fetching earnings call data for {symbol}, {year}, Q{quarter}")
        return []

# Test the function
response = get_earnings_call("AAPL", 2020, "Q3")
print("Response length:", len(response))
print("Response data:", response[:2] if response else "No data")  