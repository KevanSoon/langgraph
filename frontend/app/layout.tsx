import type { Metadata } from 'next'

import './globals.css'
import { ThemeProvider } from '../context/ThemeContext';

export const metadata: Metadata = {
  title: 'Exypnos',
  description: 'Created with v0',
  generator: 'v0.dev',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>
        <ThemeProvider>
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
