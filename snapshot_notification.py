from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('http://localhost:8502')
    
    print("Logging in...")
    page.fill('input[type=\"text\"]', 'supriyakambali@ibm.com')
    page.fill('input[type=\"password\"]', 'manager123')
    page.get_by_text('Sign in →').click()
    
    print("Waiting for dashboard...")
    page.wait_for_timeout(3000)
    
    print("Opening notifications popover...")
    page.get_by_test_id('stPopoverButton').first.click()
    
    print("Capturing screenshot...")
    page.wait_for_timeout(1000)
    
    page.screenshot(path='/Users/supriyapkambali/Documents/Week4/Deliverables/IBM_DeliveryIQ/notification_popover.png')
    print("Done!")
    
    browser.close()
