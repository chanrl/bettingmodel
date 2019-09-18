from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
import selenium.webdriver.support.ui as ui
import selenium.webdriver.support.expected_conditions as EC
import pandas as pd
import os
import time
import glob
import re

# page = pd.read_html('http://www.sportsplays.com/consensus/all.html')

# http://www.sportsplays.com/consensus/NFL.html

# page2  = pd.read_html('https://www.oddsshark.com/stats/dbresults/football/nfl')

# https://www.oddsshark.com/nfl/database

# page[0].iloc[:,[1]]

class odds():
  def __init__(self):
    options = webdriver.ChromeOptions()
    self.driver = webdriver.Chrome(options=options)

  def gotoodds(self,team):
    self.driver.get('https://www.oddsshark.com/nfl/database')
    ui.WebDriverWait(self.driver, 15).until(EC.visibility_of_element_located((By.ID, "team-search-h2h")))
    self.driver.find_element_by_id("team-search-tab").send_keys(team)
    self.driver.find_element_by_id("games-30-h2h").click()
    self.driver.find_element_by_id("chalk-select-game-type-h2h").click()
    self.driver.find_element_by_id("location-any-h2h").click()
    self.driver.find_element_by_id("chalk-select-odds-h2h").click()
    self.driver.find_element_by_id("submit-h2h").click()

  def teardown(self):
    self.driver.close()


    #html = driver.page_source