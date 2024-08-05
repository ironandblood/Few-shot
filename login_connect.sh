#!/bin/bash

username="20307130261"
password="Xc142857142857"

URL="https://10.108.255.249/include/auth_action.php"

# curl $URL --insecure --data "action=login&username=$username&password=$password&ac_id=1&user_ip=&nas_ip=&user_mac=&save_me=1&ajax=1" > /dev/null 2>&1
curl $URL --insecure --data "action=login&username=$username&password=$password&ac_id=1&user_ip=&nas_ip=&user_mac=&save_me=1&ajax=1"