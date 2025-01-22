import matplotlib.pyplot as plt
import pandas as pd

# Sample data: Load your actual data from above as a dataframe.
data = {
    'Year': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
    'Population': [282162411, 284968955, 287625193, 290326418, 293655405, 296507061, 299398484, 301231207, 304093966, 306771529, 308745538, 311583481, 313914040, 316128839, 318857056, 321418820, 324459463, 327167439, 329484123, 331449000, 331449000]  # Example values
}

df = pd.DataFrame(data)

# Create line chart
plt.plot(df['Year'], df['Population'], marker='o')
plt.title('US Population Trend')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.show()