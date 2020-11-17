import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from rhymes.items import RhymesItem, RhymesLoader


class RhymedbSpider(CrawlSpider):
    name = "rhymedb"
    allowed_domains = ["rhymedb.com"]
    start_urls = ["http://rhymedb.com/"]

    rules = (
        Rule(LinkExtractor(allow=r"/az/\w"), callback=None),
        # Rule(LinkExtractor(allow=r"/az/a"), callback=None),
        Rule(
            LinkExtractor(allow=r"/word/\w+"),
            # LinkExtractor(allow=r"/word/cabals"),
            callback="parse_index_page",
        ),
        # Rule(LinkExtractor(allow=r'/az/\w'), callback='parse_index_page', follow=True),
    )

    def parse_index_page(self, response):
        item = RhymesLoader(item=RhymesItem(), response=response)
        item.add_value("word", response.url.split('/')[-1])
        rhymes = item.nested_xpath('//span[contains(@class, "content")]')
        rhymes.add_xpath("rhymes2", './a[contains(@class, "top2")]/text()')
        rhymes.add_xpath("rhymes1", './a[contains(@class, "top1")]/text()')
        rhymes.add_xpath(
            "rhymes3",
            './a[not(contains(@class, "top2") or contains(@class, "top1"))]/text()',
        )
        # item.add_xpath('rhymes1', '//span[contains(@class, "content")]/a/text()')
        return item.load_item()
