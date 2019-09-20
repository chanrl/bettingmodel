from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import selenium.webdriver.support.ui as ui
import selenium.webdriver.support.expected_conditions as EC
import pandas as pd
import time

# page = pd.read_html('http://www.sportsplays.com/consensus/all.html')

# http://www.sportsplays.com/consensus/NFL.html

# https://www.oddsshark.com/nfl/database

class scrape():
  def __init__(self):
    options = webdriver.ChromeOptions()
    self.driver = webdriver.Chrome(options=options)

  def oddsshark(self,team=""):
    '''
    Parameters
    ----------
    team = str of team e.g.'Atlanta'
    '''
    self.driver.get('https://www.oddsshark.com/nfl/database')
    ui.WebDriverWait(self.driver, 5).until(EC.visibility_of_element_located((By.ID, "team-search-h2h")))
    self.driver.find_element_by_id("team-search-h2h").send_keys(team)
    self.driver.find_element_by_xpath('//*[@id="frm-h2h"]/table/tbody/tr[3]/td/div/div[3]/label').click()
    self.driver.find_element_by_xpath('//*[@id="frm-h2h"]/table/tbody/tr[6]/td/div/div[1]/label').click()
    self.driver.execute_script('document.getElementById("chalk-select-game-type-h2h").value = "REG/PST"')
    self.driver.execute_script('document.getElementById("chalk-select-odds-h2h").value = "ANY"')
    time.sleep(3)
    self.driver.find_element_by_xpath('//*[@id="submit-h2h"]').click()
    time.sleep(3)
    self.page = self.driver.page_source
    table = pd.read_html(self.page)
    table[1].to_csv(f'data/{team}.csv')
    self.driver.back()

  def sportsplays(self, month, day, year):
    '''
    Parameters
    ----------
    month - str of month e.g December
    day - int 2 digits e.g. 01
    year - int 4 digits e.g. 2019
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
    table[0].to_csv(f'data/{month}-{day}-{year}.csv')
    self.driver.back()

  def quit(self):
    self.driver.close()