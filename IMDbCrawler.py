import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

# movie = 'http://www.imdb.com/title/tt2527336/'
# html = requests.get(movie)           # connect to the server
#
# bs = BeautifulSoup(html.text, 'lxml')        # manage tags in HTML document
#
# # crawl movie description
# desc = bs.find('div', {'class': 'summary_text'})
# movie_desc = desc.get_text().strip()
# print(movie_desc)
#
# # crawl the poster
# poster = bs.find('div', {'class': 'poster'})
# img = poster.find('img')
# img_link = img['src']       # access the value of an attribute of a tag print(img_link)
# image = requests.get(img_link)
# with open('star_war_poster.jpg', 'wb') as f:
#     f.write(image.content)      # writing the picture
#
# ## find more
# recommended_movies = bs.find('div', {'class': 'rec_page'})
# movie_list = recommended_movies.find_all('div', {'class': 'rec_item'})
# for ml in movie_list:
#     # print(ml)
#     movie_id = ml['data-tconst']    ##movie_id saved in the attribute called data-tconst
#     movie_link = 'http://www.imdb.com/title/' + movie_id + '/'
#     print(movie_link)

website = 'https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm'
html = requests.get(website)  # connect to the server
bs = BeautifulSoup(html.text, 'lxml')
table = bs.find('tbody', {'class': 'lister-list'})
tr_list = table.find_all('tr')
output = []
for tr in tr_list:
    movie = tr.find('td', {'class': 'titleColumn'})
    # print(movie, "\n")

    # find the title
    link = movie.find("a")
    title = link.get_text()
    print("title:", title)

    # find the release year
    span = movie.find("span")
    release_year = span.get_text()
    release_year = re.sub('[;:.,!?\-/+^\'_$%*()`~\"@#&={}\[\]|\\\\<>]', '', release_year)  # remove the symbol
    print("release_year:", release_year)

    ## find the ranking
    rank = movie.find("div", {"class": "velocity"})
    up_or_down = rank.get_text()
    if "no change" in up_or_down:
        up_or_down = "no change"
    else:
        rank = rank.find("span", {"class", "global-sprite"})
        up_or_down = rank['class'][2]
        # print(rank['class'][2])
    print("up_or_down:", up_or_down)

    ## find the overall rating
    td = tr.find('td', {'class': 'ratingColumn imdbRating'})
    strong = td.find('strong')
    if strong is not None:
        rating = strong.get_text()
    else:
        rating = "None"
    print("rating:", rating)
    list = [title, release_year, up_or_down, rating]
    output.append(list)

    ## get the website of each movie
    td = tr.find('div', {'class': 'seen-widget'})
    movie_id = td['data-titleid']
    movie_link = 'http://www.imdb.com/title/' + movie_id + '/'
    print(movie_link)
    each_movie = requests.get(movie_link)
    new_page = BeautifulSoup(each_movie.text, 'lxml')

    # crawl movie description
    desc = new_page.find('div', {'class': 'summary_text'})
    movie_desc = desc.get_text().strip()
    print("summary text: ", movie_desc)

    # crawl the poster
    poster = new_page.find('div', {'class': 'poster'})
    img = poster.find('img')
    img_link = img['src']       # access the value of an attribute of a tag print(img_link)
    image = requests.get(img_link)
    page_name = title + '.jpg'
    with open(title + '.jpg', 'wb') as f:
        f.write(image.content)      # writing the picture
        print(title + '.jpg' + ' saved!' + "\n")



print(output)
name = ['title', 'release year', 'ranking', 'rating']
test = pd.DataFrame(data=output, columns=name)
# print(test)
test.to_csv('output.csv', index=False)





