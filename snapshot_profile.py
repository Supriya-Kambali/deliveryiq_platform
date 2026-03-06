from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto('http://localhost:8502')
    
    print("Logging in...")
    page.fill('input[placeholder="Enter username"]', 'manager')
    page.fill('input[placeholder="Enter password"]', 'manager123')
    page.click('button:has-text("Sign in →")')
    
    print("Waiting for dashboard...")
    page.wait_for_selector('text=Operations Dashboard', timeout=10000)
    page.wait_for_timeout(2000)
    
    print("Opening popover...")
    page.click('button:has-text("👤 M")')
    
    print("Capturing screenshot...")
    page.wait_for_selector('text=manager@ibm.com', timeout=5000)
    page.wait_for_timeout(1000)
    
    page.screenshot(path='profile_popover.png')
    print("Done!")
    
    browser.close()
