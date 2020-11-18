# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.item import Item, Field
from scrapy.loader import ItemLoader
from itemloaders.processors import MapCompose, TakeFirst


class RhymesItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    word = scrapy.Field()
    rhymes1 = scrapy.Field()
    rhymes2 = scrapy.Field()
    rhymes3 = scrapy.Field()

class RhymesLoader(ItemLoader):
    """Process an item before pipeline."""
    word_in = MapCompose(str.strip)
    word_out = TakeFirst()
    rhymes1_in = MapCompose(str.lower)
    rhymes2_in = MapCompose(str.lower)
    rhymes3_in = MapCompose(str.lower)
