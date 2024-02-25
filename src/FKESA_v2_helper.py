#!/usr/bin/env python3
#Author: Pratik M Tambe <enthusiasticgeek@gmail.com>
#Date: Jan 20, 2024
#Helper functions

import requests

class FKESAHelper:
    def __init__(self):
        # You can initialize any class-specific variables here
        pass

    def get_boolean_value_from_url(self, url, timeout_seconds=5):
        """
        Sends an HTTP GET request to the specified URL and converts the response to a boolean value.

        Parameters:
        - url (str): The URL to send the GET request to.
        - timeout_seconds (int): Timeout for the HTTP request in seconds. Default is 5 seconds.

        Returns:
        - bool or None: The boolean value obtained from the response content. Returns None in case of an error.
        """
        try:
            # Send GET request to the server with a specified timeout
            response = requests.get(url, timeout=timeout_seconds)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Convert the response content (string) to a boolean value
                lowercased_response = response.text.lower()
                if lowercased_response == "true":
                    return True
                elif lowercased_response == "false":
                    return False
                else:
                    print(f"Error: Unexpected response content: {response.text}")

            else:
                print(f"Error: Unable to fetch data. Status Code: {response.status_code}")

        except requests.ConnectionError:
            print("Error: Connection to the server failed.")
        except requests.RequestException as e:
            print(f"Error: {e}")
        except requests.Timeout:
            print(f"Error: Request timed out after {timeout_seconds} seconds.")

        # Return None if an error occurs
        return None

    def get_int_value_from_url(self, url, timeout_seconds=5):
        """
        Sends an HTTP GET request to the specified URL and converts the response to an integer value.

        Parameters:
        - url (str): The URL to send the GET request to.
        - timeout_seconds (int): Timeout for the HTTP request in seconds. Default is 5 seconds.

        Returns:
        - int or None: The integer value obtained from the response content. Returns None in case of an error.
        """
        try:
            # Send GET request to the server with a specified timeout
            response = requests.get(url, timeout=timeout_seconds)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Convert the response content (string) to an integer value
                try:
                    int_value = int(response.text)
                    return int_value
                except ValueError:
                    print(f"Error: Unable to convert response to integer: {response.text}")

            else:
                print(f"Error: Unable to fetch data. Status Code: {response.status_code}")

        except requests.ConnectionError:
            print("Error: Connection to the server failed.")
        except requests.RequestException as e:
            print(f"Error: {e}")
        except requests.Timeout:
            print(f"Error: Request timed out after {timeout_seconds} seconds.")

        # Return None if an error occurs
        return None

    def get_string_value_from_url(self, url, timeout_seconds=5):
        """
        Sends an HTTP GET request to the specified URL and returns the response content as a string.

        Parameters:
        - url (str): The URL to send the GET request to.
        - timeout_seconds (int): Timeout for the HTTP request in seconds. Default is 5 seconds.

        Returns:
        - str or None: The string value obtained from the response content. Returns None in case of an error.
        """
        try:
            # Send GET request to the server with a specified timeout
            response = requests.get(url, timeout=timeout_seconds)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Return the response content as a string
                return response.text

            else:
                print(f"Error: Unable to fetch data. Status Code: {response.status_code}")

        except requests.ConnectionError:
            print("Error: Connection to the server failed.")
        except requests.RequestException as e:
            print(f"Error: {e}")
        except requests.Timeout:
            print(f"Error: Request timed out after {timeout_seconds} seconds.")

        # Return None if an error occurs
        return None

    def get_float_value_from_url(self, url, timeout_seconds=5):
        """
        Sends an HTTP GET request to the specified URL and converts the response to a float value.

        Parameters:
        - url (str): The URL to send the GET request to.
        - timeout_seconds (int): Timeout for the HTTP request in seconds. Default is 5 seconds.

        Returns:
        - float or None: The float value obtained from the response content. Returns None in case of an error.
        """
        try:
            # Send GET request to the server with a specified timeout
            response = requests.get(url, timeout=timeout_seconds)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Convert the response content (string) to a float value
                try:
                    float_value = float(response.text)
                    return float_value
                except ValueError:
                    print(f"Error: Unable to convert response to float: {response.text}")

            else:
                print(f"Error: Unable to fetch data. Status Code: {response.status_code}")

        except requests.ConnectionError:
            print("Error: Connection to the server failed.")
        except requests.RequestException as e:
            print(f"Error: {e}")
        except requests.Timeout:
            print(f"Error: Request timed out after {timeout_seconds} seconds.")

        # Return None if an error occurs
        return None

    def post_data_to_url(self, url, data, timeout_seconds=5):
        """
        Sends an HTTP POST request to the specified URL with the given data.

        Parameters:
        - url (str): The URL to send the POST request to.
        - data (dict): The data to be sent in the POST request.
        - timeout_seconds (int): Timeout for the HTTP request in seconds. Default is 5 seconds.

        Returns:
        - requests.Response or None: The response object obtained from the POST request.
          Returns None in case of an error.
        """
        try:
            # Send POST request to the server with a specified timeout
            response = requests.post(url, data=data, timeout=timeout_seconds)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                return response
            else:
                print(f"Error: Unable to post data. Status Code: {response.status_code}")

        except requests.ConnectionError:
            print("Error: Connection to the server failed.")
        except requests.RequestException as e:
            print(f"Error: {e}")
        except requests.Timeout:
            print(f"Error: Request timed out after {timeout_seconds} seconds.")

        # Return None if an error occurs
        return None

"""
# Example usage:
url_boolean = "http://192.168.4.1/handleReachedBeginX"
url_integer = "http://example.com/getIntegerValue"
url_string = "http://example.com/getStringValue"
url_float = "http://example.com/getFloatValue"

# Create an instance of FKESA_v2_helper
helper = FKESAHelper()

# Call the boolean method on the instance
result_boolean = helper.get_boolean_value_from_url(url_boolean)
if result_boolean is not None:
    print("Received boolean value:", result_boolean)

# Call the integer method on the instance
result_integer = helper.get_int_value_from_url(url_integer)
if result_integer is not None:
    print("Received integer value:", result_integer)

# Call the string method on the instance
result_string = helper.get_string_value_from_url(url_string)
if result_string is not None:
    print("Received string value:", result_string)

# Call the float method on the instance
result_float = helper.get_float_value_from_url(url_float)
if result_float is not None:
    print("Received float value:", result_float)


# Example usage:
url_post = "http://192.168.4.1/button2"
data_post = {"textbox2": "500"}

# Call the post_data method on the instance
response_post = helper.post_data_to_url(url_post, data_post)
if response_post is not None:
    print("POST Request Response:")
    print(response_post.text)
    print("Headers:")
    print(response_post.headers)

"""
