import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import matplotlib.pyplot as plt

#Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Base URL of the website
url = "https://books.toscrape.com/catalogue/page-{}.html"
base_url = "https://books.toscrape.com/catalogue/"

# Create directories if they don't exist
os.makedirs('data_sheet', exist_ok=True)
os.makedirs('images', exist_ok=True)

# List to store book details
books = []
#print(repr(base_url)) Used to debug

# Function to extract data from a single page
def extract_data_from_page(soup, page):
    for book in soup.find_all('article', class_='product_pod'):
        title = book.h3.a['title']
        price = book.find('p', class_='price_color').text
        availability = book.find('p', class_='instock availability').text.strip()
        rating_text = book.p['class'][1]
        rating = convert_rating_to_number(rating_text)
        link = base_url + book.h3.a['href']
        #print("BASE URL =", repr(base_url)) Used to debug
        #print("HREF =", repr(book.h3.a['href'])) Used to debug
        #print("HREF =", repr(base_url + book.h3.a['href'])) Used to debug
        #print("LINK VALUE:", repr(link)) Used to debug
        #print("LINK VALUE:", repr(link)) Used to debug
        img_src = book.find('img')['src']
        #print("IMG SRC:", img_src) Used to debug
        thumbnail_url = ('https://books.toscrape.com/' + img_src.replace('../',''))
        #print("THUMBNAIL:", thumbnail_url) Used to debug
        thumbnail_file_name = save_thumbnail_image(thumbnail_url, title)

        book_data = {
            'Title': title,
            'Price': price,
            'Availability': availability,
            'Rating': rating,
            'Link': link,
            'Thumbnail URL': thumbnail_url,
            'Thumbnail File Name': thumbnail_file_name,
        }

        books.append(book_data)
        #print(link) Used to debug
        #print(type(link)) Used to debug
        #print(book.h3.a['href']) Used to debug


        # Print the extracted data to the console in a formatted way
        print(f"Page: {page}")
        print(f"Title: {title}")
        print(f"Price: {price}")
        print(f"Availability: {availability}")
        print(f"Rating: {rating}")
        print(f"Link: {link}")
        print(f"Thumbnail URL: {thumbnail_url}")
        print(f"Thumbnail File Name: {thumbnail_file_name}")
        print("="*50)

# Function to convert rating text to a number
def convert_rating_to_number(rating_text):
    rating_dict = {
        'One': 'One Star',
        'Two': 'Two Star',
        'Three': 'Three Star',
        'Four': 'Four Star',
        'Five': 'Five Star',
    }
    return rating_dict.get(rating_text, 0)

# Function to save thumbnail image
def save_thumbnail_image(url, title):
    response = requests.get(url)
    sanitized_title = "".join([c if c.isalnum() else "_" for c in title])
    image_path = os.path.join('images', f"{sanitized_title}.jpg")
    with open(image_path, 'wb') as f:
        f.write(response.content)
    return image_path

# Loop through the first 10 pages
for page in range(1, 11):
    page_url = url.format(page)
    response = requests.get(page_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    extract_data_from_page(soup, page)
    print(f"Extracted data from page: {page}")

# Save the data to a CSV file
df = pd.DataFrame(books)
df['Price'] = df['Price'].str.replace('£','', regex=False) #stage 2
df['Price'] = pd.to_numeric(df['Price']) #stage 2
print("Duplicate rows:", df.duplicated().sum()) #stage 2
print(df['Rating'].value_counts()) #stage 2
print(df.shape)
print("Total books scraped:", len(books))
#print("BASE_URL:", repr(base_url)) Used to debug
#print(repr(df['Link'][0])) Used to debug
#print(repr(df['Thumbnail URL'][0])) Used to debug
#print(df['Link'].head()) Used to debug
print(df.isnull().sum())
print(df.dtypes) #stage 2
print(df.describe()) #stage 2

#Most Expensive Book
most_expensive = df.loc[df['Price'].idxmax()] #stage 3
print("Most Expensive Book Being:") #stage 3
print(most_expensive[['Title','Price']]) #stage 3

# Cheapest Book
cheapest = df.loc[df['Price'].idxmin()] #stage 3
print("Cheapest Book Being:") #stage 3
print(cheapest[['Title','Price']]) #stage 3

# Average Price by Rating
print(df.groupby('Rating')['Price'].mean()) #stage 3

# Number of Books per Rating
print(df['Rating'].value_counts()) #stage 3

#Most Expensive Books List
top = df.nlargest(5, 'Price') #stage 3
print(top[['Title','Price']]) #stage 3

# Visualization of Ratings Distribution Stage 4
df['Rating'].value_counts().sort_index().plot(
    kind = 'bar',
    title = 'Book Ratings of Distribution'
)
plt.xlabel('Rating')
plt.ylabel('Number of Books')
plt.show()

# Visualization of Price Distribution
df['Price'].plot(
    kind = 'hist',
    bins = 10,
    title = 'Distribution of Books Prices'
)
plt.xlabel('Price (£)')
plt.show()

# Average Price by Rating
df.groupby('Rating')['Price'].mean().plot(
    kind = 'bar',
    title = 'The Average Price by Rating'
)
plt.xlabel('Rating')
plt.ylabel('Average Price (£)')
plt.show()

plt.scatter(df['Rating'],df['Price'])
plt.title('Price vs Rating')
plt.xlabel('Rating')
plt.ylabel('Price (£)')
plt.show()

print(df.duplicated().sum()) #stage 2

# Linear Regression
X = df[['Rating']]
Y = df['Price']
x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size = 0.2,
    random_state = 42,
)
model = LinearRegression()
model.fit(x_train, y_train)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Training rows:", len(x_train))
print("Testing rows:", len(x_test))
prediction = model.predict(x_test)
print("R² Score:", r2_score(y_test, prediction))
print("Mean Absolute Errors:", mean_absolute_error(y_test, prediction))
print("Mean Squared Error:", mean_squared_error(y_test, prediction))
prediction = model.predict(x_test)

plt.scatter(y_test, prediction)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color = 'red',
)

plt.xlabel("The Actual Price")
plt.ylabel("The Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

df.to_csv('data_sheet/books.csv', index=False)
df.to_csv('data_sheet/books_cleaned.csv', index=False)
print("Data saved to data_sheet/books_data.csv")
