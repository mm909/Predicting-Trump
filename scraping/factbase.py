import re
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options

print("Starting Chrome")
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome("E:/chromedriver/chromedriver.exe",options=chrome_options)
driver.get('https://factba.se/transcripts')
actions = ActionChains(driver)

speechTimeline = driver.find_element_by_xpath("/html/body/div[3]/div[1]/div[2]/div[1]/div[1]/ul")
listOfSpeeches = speechTimeline.find_elements_by_tag_name("li")
for Speech in listOfSpeeches:
    if Speech.get_attribute("style") == 'margin-bottom: 120px;':
        listOfSpeeches.remove(Speech)
    pass
print("Number of speeches: " + str(len(listOfSpeeches)))

speech = listOfSpeeches[1]
actions.move_to_element(speech).perform()
item = speech.find_element_by_xpath("./div[2]")
item.click()
time.sleep(2)
# print(item.get_attribute("class"))
# item = item.find_element_by_xpath("./h3")
# actions.moveToElement(element).click().perform();`
# driver.execute_script("arguments[0].click();", item)

title = driver.find_element_by_xpath("/html/body/div[2]/div/div[1]/div/span[1]/h1").text
print("Reading: " + title)
title = re.sub(r'\W+', '', title)

print("Looking for Trump...")
item = driver.find_elements_by_xpath("//*[@id='resultsblock']/div")
for listItem in item:
    if listItem.get_attribute("class") == 'media topic-media-row mediahover not-trump':
        item.remove(listItem)
    pass

print("Reading text...")
text = ""
for listItem in item:
    text = text + listItem.find_element_by_xpath("./div[3]/div/div[2]/a").text + " "

print("Writting text...")
tc = open("../data/trump/speeches/raw/factbase/"+title+".txt","w", encoding="utf8")
tc.write(text)

driver.execute_script("window.history.go(-1)");

print("Cleaning up...")
# time.sleep(5)
driver.quit()
