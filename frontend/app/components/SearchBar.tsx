'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

const smartSchemes = ['ltc', 'lnc', 'ntc', 'anc'] // or pull this from a config

export default function SearchBar() {
  const [query, setQuery] = useState('')
  const [queryScheme, setQueryScheme] = useState('ntc') // default
  const [docScheme, setDocScheme] = useState('ntc')     // default
  const router = useRouter()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Navigate to: /?q=...&query_scheme=...&doc_scheme=...
    const params = new URLSearchParams({
      q: query,
      query_scheme: queryScheme,
      doc_scheme: docScheme,
    })
    router.push(`/?${params.toString()}`)
  }

  return (
    <form onSubmit={handleSubmit}>
      {/* First row: search input + submit button */}
      <div className="flex gap-2 mb-6 items-center">
        <Input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search documents..."
          className="flex-grow"
        />
        <Button type="submit">Search</Button>
      </div>

      {/* Second row: Query Scheme button group */}
      <div className="mb-4">
        <h3 className="text-sm font-medium mb-2">Query Scheme</h3>
        <div className="flex gap-2">
          {smartSchemes.map((scheme) => {
            const isSelected = scheme === queryScheme
            return (
              <button
                key={scheme}
                type="button"
                onClick={() => setQueryScheme(scheme)}
                className={`
                  px-3 py-1 rounded 
                  ${isSelected ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'}
                  transition-colors
                `}
              >
                {scheme.toUpperCase()}
              </button>
            )
          })}
        </div>
      </div>

      {/* Third row: Document Scheme button group */}
      <div className="mb-4">
        <h3 className="text-sm font-medium mb-2">Document Scheme</h3>
        <div className="flex gap-2">
          {smartSchemes.map((scheme) => {
            const isSelected = scheme === docScheme
            return (
              <button
                key={scheme}
                type="button"
                onClick={() => setDocScheme(scheme)}
                className={`
                  px-3 py-1 rounded 
                  ${isSelected ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700'}
                  transition-colors
                `}
              >
                {scheme.toUpperCase()}
              </button>
            )
          })}
        </div>
      </div>
    </form>
  )
}

