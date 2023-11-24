import requests
import os

# for every line in output.txt, i want to download a file of the form url + (line) +   .pdf

# open the file
f = open("output.txt", "r")
# read the file
lines = f.readlines()
# close the file
f.close()

# for every line in the file
for line in lines:
    #get last word in line
    name = line.split("/")[-2].strip()
    # get the url 
    url = "http://collegecatalog.uchicago.edu" + line.strip() + name + ".pdf"
    print(url)
    # get the file name
    file_name = name + ".pdf"
    # make a get request
    response = requests.get(url)
    # check if the request was successful
    if response.status_code == 200:
        # if it was, write the file
        with open(file_name, 'wb') as f:
            f.write(response.content)
    else:
        # if it wasn't, print an error message
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
