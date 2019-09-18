from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import selenium.webdriver.support.ui as ui
import selenium.webdriver.support.expected_conditions as EC
import pandas as pd
import os
import time
import glob
import re

# page = pd.read_html('http://www.sportsplays.com/consensus/all.html')

# http://www.sportsplays.com/consensus/NFL.html

# https://www.oddsshark.com/nfl/database

class scrape():
  def __init__(self):
    options = webdriver.ChromeOptions()
    self.driver = webdriver.Chrome(options=options)

  def oddsshark(self,team=""):
    self.driver.get('https://www.oddsshark.com/nfl/database')
    ui.WebDriverWait(self.driver, 5).until(EC.visibility_of_element_located((By.ID, "team-search-h2h")))
    self.driver.find_element_by_id("team-search-h2h").send_keys(team)
    try:
      self.driver.find_element_by_id("games-30-h2h").click()
    except:
      print("games-30 click doesn't work")
    try:
      dataset_drop_down_element = ui.WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.ID, 'chalk-select-game-type-h2h')))
      dataset_drop_down_element = Select(dataset_drop_down_element)
      #dataset_drop_down_element.select_by_visible_text('Regular Season')
      dataset_drop_down_element.select_by_index(1) 
    except:
      print("Played in drop down doesn't work")
    try:  
      self.driver.find_element_by_id("location-any-h2h").click()
    except:
      print("Anywhere button doesn't work")
    try:
      dataset_drop_down_element = ui.WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.ID, 'chalk-select-odds-h2h')))
      dataset_drop_down_element = Select(dataset_drop_down_element)
      dataset_drop_down_element.select_by_visible_text('Either')
    except:
      print("favorite/underdog dropdown doesn't work")
    try:
      self.driver.find_element_by_id("submit-h2h").click() #working
    except:
      print("search button doesn't work")
    #page = driver.page_source

  def sportsplays(self, month, day, year):
    '''
    Parameters
    ----------
    month - str of month e.g December
    '''
    self.driver.get('http://www.sportsplays.com/consensus/all.html')
    dataset_drop_down_element = ui.WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.ID, 'from_date_month')))
    dataset_drop_down_element = Select(dataset_drop_down_element)
    dataset_drop_down_element.select_by_visible_text(month)
    dataset_drop_down_element = ui.WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.ID, 'from_date_day')))
    dataset_drop_down_element = Select(dataset_drop_down_element)
    dataset_drop_down_element.select_by_visible_text(day)
    dataset_drop_down_element = ui.WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.ID, 'from_date_year')))
    dataset_drop_down_element = Select(dataset_drop_down_element)
    dataset_drop_down_element.select_by_visible_text(year)
    self.driver.find_element_by_name('commit').click()
    self.page = self.driver.page_source
    table = pd.read_html(self.page)
    return table

  def quit(self):
    self.driver.close()