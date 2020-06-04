from abc import ABCMeta, abstractmethod
class Book(object, metaclass=ABCMeta):
    def __init__(self,title,author):
        self.title=title
        self.author=author   
    @abstractmethod
    def display(): pass

#Write MyBook class
class MyBook(Book):
    def __init(self, title, author, price):
        self.price = price
        Book.__init__(self, title, author)
    def display(self):
        print('Title: {}'.format(self.title))
        print('Author: {}'.format(self.author))
        print('Price: {}'.format(self.price))

new_novel=MyBook('a', 'b', 13)
new_novel.display()