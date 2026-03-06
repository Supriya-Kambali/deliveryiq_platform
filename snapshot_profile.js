const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  
  console.log("Navigating to app...");
  await page.goto('http://localhost:8502');
  
  console.log("Logging in as manager...");
  await page.fill('input[placeholder="Enter username"]', 'manager');
  await page.fill('input[placeholder="Enter password"]', 'manager123');
  await page.click('button:has-text("Sign in →")');
  
  // Wait for dashboard to load
  await page.waitForSelector('text=Operations Dashboard', { timeout: 10000 });
  await page.waitForTimeout(2000); // let charts render
  
  console.log("Opening profile popover...");
  // Find the button with the User Profile emoji "👤 M"
  await page.click('button:has-text("👤 M")');
  
  // Wait for the popover content to appear
  await page.waitForSelector('text=manager@ibm.com', { timeout: 5000 });
  await page.waitForTimeout(1000);
  
  await page.screenshot({ path: 'profile_popover.png' });
  console.log("Screenshot saved to profile_popover.png");
  
  await browser.close();
})();
