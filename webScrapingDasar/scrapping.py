from selenium.webdriver.firefox.options import Options
#from selenium.webdriver.firefox.service import Service
from selenium import webdriver
import time
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
import re
import pandas as pd

# SETUP FIREFOX
options = Options()
#options.headless = True # Jalankan tanpa tampilan GUI
#service = Service() # geckodriver otomatis ditemukan jika sudah di PATH

# Jalankan Firefox driver
driver = webdriver.Firefox(options=options)

# Ganti URL dengan web yang ingin di-scrape
url = "https://www.imdb.com/title/tt3514324/reviews/?ref_=tt_ururv_sm"
driver.get(url)
time.sleep(10) # Tunggu render halaman

# Scroll dan klik "Load More" jika ada
x = 1
while True:
    try:
        load_more_button = driver.find_element(By.CSS_SELECTOR, 'button.ipc-see-more__button')
        driver.execute_script("arguments[0].click();", load_more_button)
        print(f"Load ke-{x}")
        x = x+1
        time.sleep(2)
    except NoSuchElementException:
        print("Semua review sudah dimuat.")
        break


# Ambil elemen review
review_blocks = driver.find_elements(By.CSS_SELECTOR, 'article[class*="user-review-item"]')
print(f"Total review ditemukan: {len(review_blocks)}")

data = []

for review in review_blocks:
    try:
        title = driver.find_element(By.CSS_SELECTOR, 'h2[data-testid="subtitle"]').text.strip()
    except:
        title = '-'
    
    try:
        user = review.find_element(By.CSS_SELECTOR, 'a[data-testid="author-link"]').text.strip()
    except:
        user = '-'
    
    try:
        tanggal = review.find_element(By.CSS_SELECTOR, 'li.review-date').text.strip()
    except:
        tanggal = '-'
    
    try:
        rating = review.find_element(By.CSS_SELECTOR, 'span.ipc-rating-star--rating').text.strip()
    except:
        rating = '-'

    try:
        like = review.find_element(By.CSS_SELECTOR, 'span.ipc-voting__label__count--up').text.strip()
    except:
        like = '-'

    try:
        dislike = review.find_element(By.CSS_SELECTOR, 'span.ipc-voting__label__count--down').text.strip()
    except:
        dislike = '-'

    try:
        title_comment = review.find_element(By.CSS_SELECTOR, 'h3.ipc-title__text.ipc-title__text--reduced').text.strip()
    except:
        title_comment = '-'
    
    try:
        comment = review.find_element(By.CSS_SELECTOR, 'div[data-testid="review-content"]').text.strip()
    except:
        try:
            comment = review.find_element(By.CSS_SELECTOR, 'div.ipc-html-content-inner-div').text.strip()
        except:  
            comment = '-'

    data.append({
        'Nama Reviewer': user,
        'Tanggal': tanggal,
        'Judul Review': title,
        'Rating': rating,
        'Like' : like,
        'Dislke' : dislike,
        'Title Komentar' : title_comment,
        'Komentar': comment
    })

# Tutup browser
driver.quit()

df = pd.DataFrame(data)
df.to_excel("data_scrapping.xlsx", index=False)
# df.to_csv("data_scrapping.csv", index=False, encoding="utf-8") #kalau ingin data convert ke csv
print(" Data review berhasil disimpan ke file Excel!")