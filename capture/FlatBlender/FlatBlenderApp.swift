import SwiftUI

@main
struct FlatBlenderApp: App {
    @State private var store = ProjectStore()

    var body: some Scene {
        WindowGroup {
            ProjectsView()
                .environment(store)
        }
    }
}
