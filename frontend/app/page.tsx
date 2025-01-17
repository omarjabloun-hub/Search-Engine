export const dynamic = 'force-dynamic';

import SearchBar from './components/SearchBar'
import SearchResults from './components/SearchResults'

export default function Home({
  searchParams,
}: {
  searchParams: { q?: string; query_scheme?: string; doc_scheme?: string }
}) {
  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8 text-center">Document Search</h1>
      <SearchBar />
      <SearchResults searchParams={searchParams} />
    </main>
  )
}

