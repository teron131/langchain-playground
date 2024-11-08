import ast
import re

from tabulate import tabulate


def format_query(query: str) -> str:
    """Format SQL query with proper line breaks and indentation.

    Takes a raw SQL query string and formats it with consistent capitalization,
    line breaks, and indentation for better readability.

    Args:
        query (str): Raw SQL query string to format

    Returns:
        str: Formatted SQL query with:
            - Keywords capitalized
            - Each major clause on new line
            - JOIN...ON clauses indented
            - Normalized whitespace
    """
    keywords = ["SELECT", "FROM", "JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET"]
    query = " ".join(query.strip().split())  # Normalize whitespace

    # Convert keywords to uppercase and add line breaks
    for keyword in keywords:
        query = re.sub(rf"\b{keyword}\b", keyword, query, flags=re.IGNORECASE)
        query = query.replace(f" {keyword} ", f"\n{keyword} ")

    # Special handling for JOIN...ON
    query = query.replace(" ON ", "\n    ON ")

    return query


def data_to_table(query: str, data: str) -> str:
    """Convert SQL query results to a formatted table.

    Takes a SQL query string and query result data and formats them into a
    human-readable table with column headers extracted from the SELECT clause.

    Args:
        query (str): SQL query string containing SELECT statement
        data (str): Query results as string representation of list of tuples

    Returns:
        str: Formatted table string with:
            - Column headers from SELECT clause
            - Data rows from query results
            - Pretty table formatting with row indices

    Example:
        >>> query = "SELECT name, age FROM users"
        >>> data = "[('Alice', 25), ('Bob', 30)]"
        >>> print(data_to_table(query, data))
        +---+-------+-----+
        |   | name  | age |
        +---+-------+-----+
        | 0 | Alice |  25 |
        | 1 | Bob   |  30 |
        +---+-------+-----+
    """
    headers = []
    if query and isinstance(query, str):  # Check if query exists and is a string
        # Extract column headers from SELECT clause
        SELECT_clause = re.search(r"SELECT\s+(.*?)\s+FROM", query, re.IGNORECASE).group(1)
        headers = [col.strip() for col in SELECT_clause.split(",")]
        headers = [re.split(r"\s+as\s+", col.strip(), flags=re.IGNORECASE)[-1] for col in headers]

    # Parse data string into list of tuples - handle empty string case
    data_list = ast.literal_eval(data) if (isinstance(data, str) and data.strip()) else []

    return tabulate(data_list, headers, tablefmt="pretty", showindex=True)


if __name__ == "__main__":
    # Test format_query
    test_query = "select name, age from users where age > 21 order by name"
    print("\nTesting format_query:")
    print(format_query(test_query))

    # Test data_to_table with empty inputs
    print("\nTesting data_to_table with empty inputs:")
    print(data_to_table("", ""))

    # Test data_to_table with valid inputs
    test_query = "SELECT FirstName as name, LastName as surname FROM Employee"
    test_data = "[('John', 'Doe'), ('Jane', 'Smith')]"
    print("\nTesting data_to_table with valid inputs:")
    print(data_to_table(test_query, test_data))

    # Test data_to_table with valid inputs
    test_query = None
    test_data = None
    print("\nTesting data_to_table with valid inputs:")
    print(data_to_table(test_query, test_data))
