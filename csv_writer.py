import csv


class TweetCSVWriter:
    '''
    Creates a csv writer object used to extract the data to a csv file.
    '''

    def __init__(self, csv_name="", column_names=[], separator=","):
        self.csv_name = csv_name
        self.column_names = column_names
        self.separator = separator
        self.csv_file = open(self.csv_name + '.csv', 'a', encoding="utf-8")
        with self.csv_file as f:
            wr = csv.writer(f, delimiter=self.separator)
            wr.writerow(self.column_names)

    def tweet_writer(self, tweet_data):
        data_file = open(self.csv_file.name, 'a', encoding="utf-8")

        with data_file as fl:
            self.csvWriter = csv.writer(fl, delimiter=self.separator)
            try:
                self.csvWriter.writerow(tweet_data)
            except:
                print(tweet_data)
        return
